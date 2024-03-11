import {
  NDArray,
  ArtifactCache,
  detectGPUDevice,
  instantiate,
  createPolyfillWASI,
  Scalar,
} from "tvmjs";
import { Tokenizer } from "@mlc-ai/web-tokenizers";

import { DeviceNDArray, CpuNDArray, buildBias } from "./sample.js";
import { TargetDevice } from "./types.js";

import type {
  ModelSpec,
  LoadedModel,
  LoadReport,
  GenerateOptions,
  GenerationStreamHandler,
} from "./types.js";

enum ModelState {
  Waiting,
  Running,
  Cancelling,
}

const scope = (name?: string) => "ad-llama" + (name ? "/" + name : "");
const cacheScope = (name: string) => new ArtifactCache(scope(name));

const getStopIndex = (
  text: string,
  tokenDecodedText: string,
  stops: string[],
) => {
  // Check each new character in next token to see if it forms the stop sequence
  // with already completed text.
  // This gets around the issue where our stop is `"` but the next token generates `",` which
  // won't satisfy a straightforward endswith(stop)

  for (let i = tokenDecodedText.length; i >= 0; i--) {
    for (const stop of stops) {
      if (stop.length > 0 && text.slice(0, text.length - i).endsWith(stop)) {
        return text.length - i - stop.length;
      }
    }
  }

  return -1;
};

const perf = (() => {
  let entries: Record<string, number[]> = {};

  return {
    timer: (label: string) => {
      const start = performance.now();

      return () => {
        const result = performance.now() - start;
        entries[label] ??= [];
        entries[label].push(result);
        // console.debug('perf', label, result)
      };
    },
    get entries() {
      return entries;
    },
    summarize: () => {
      const sums = Object.fromEntries(
        Object.entries(entries).map(([label, results]) => [
          label,
          results.reduce((a, x) => a + x, 0),
        ]),
      );

      const averages = Object.fromEntries(
        Object.entries(sums).map(([label, sum]) => [
          label,
          sum / entries[label].length,
        ]),
      );

      console.debug("#perf", { sums, averages, entries });

      entries = {};
    },
  };
})();

export default async (
  spec: ModelSpec,
  updateReport: (loadReport: LoadReport) => void,
  targetDevice: TargetDevice,
): Promise<LoadedModel> => {
  updateReport({ loadModelConfig: "waiting" });

  const configUrl = new URL("mlc-chat-config.json", spec.modelWeightsConfigUrl)
    .href;
  const configResponse = await cacheScope("config").fetchWithCache(configUrl);
  // TODO ArtifactCache error is probably too cryptic if configurl is invalid

  const wasm = await (spec.modelLibWasmUrl.includes("localhost") // never cache localhost
    ? fetch(spec.modelLibWasmUrl)
    : cacheScope("wasm").fetchWithCache(spec.modelLibWasmUrl));

  const tvm = await instantiate(
    new Uint8Array(await wasm.arrayBuffer()),
    createPolyfillWASI(),
  );

  const device = targetDevice === TargetDevice.GPU ? tvm.webgpu() : tvm.cpu();

  if (targetDevice === TargetDevice.GPU) {
    updateReport({ detectGPU: "waiting" });

    const gpu = await detectGPUDevice();
    if (gpu == undefined) {
      updateReport({ detectGPU: "failed" });
      throw Error("Cannot find GPU in environment");
    }

    updateReport({ detectGPU: gpu.adapterInfo.vendor });

    tvm.initWebGPU(gpu.device);
  }

  let isLoadingGpuShaders = false;

  tvm.registerInitProgressCallback((report) => {
    if (isLoadingGpuShaders) {
      updateReport({ loadGPUShaders: report.progress });
    } else {
      if (report.cacheOnly) {
        updateReport({ loadModelFromCache: report.progress });
      } else {
        updateReport({ loadModelFromWeb: report.progress });
      }
    }
  });

  updateReport({ loadModel: "waiting" });

  const config = await configResponse.json();

  if (!Array.isArray(config.tokenizer_files)) {
    console.error(config);

    const err =
      'Config json file is missing an array field named "tokenizer_files"';
    updateReport({ loadModelConfig: err });
    throw Error(err);
  }

  updateReport({
    loadTokenizer: "waiting",
    loadModelConfig: "done",
  });

  const configTokenizerFiles = Object.entries({
    "tokenizer.model": Tokenizer.fromSentencePiece,
    "tokenizer.json": Tokenizer.fromJSON,
  }).find(([file, _create]) => config.tokenizer_files.includes(file));
  // preference comes from the order of tokenizer_files -- seems like .json is preferred over .model

  if (configTokenizerFiles == undefined) {
    const err = `Cant handle tokenizer files ${config.tokenizer_files}`;
    updateReport({ loadTokenizer: err });
    throw Error(err);
  }

  const [path, create] = configTokenizerFiles;

  const tokenizerResult = await cacheScope("model").fetchWithCache(
    new URL(path, spec.modelWeightsConfigUrl).href,
  );

  const tokenizer = await create(await tokenizerResult.arrayBuffer());

  updateReport({ loadTokenizer: "done" });

  await tvm.fetchNDArrayCache(
    spec.modelWeightsConfigUrl,
    device,
    scope("model"),
  );

  updateReport({ loadModel: "done" });

  tvm.beginScope();

  const vm = tvm.detachFromCurrentScope(tvm.createVirtualMachine(device));

  const prefill = tvm.detachFromCurrentScope(vm.getFunction("prefill"));

  const decode = tvm.detachFromCurrentScope(vm.getFunction("decode"));

  const getMetadata = vm.getFunction("_metadata"); // SLIM
  const metadata = JSON.parse(
    tvm.detachFromCurrentScope(getMetadata()).toString(),
  );
  console.info({ metadata });

  const stopTokens: number[] = metadata.stop_tokens ?? [2];

  const paramNames = (metadata.params as { name: string }[]).map(
    (param) => param.name,
  );

  const params = tvm.detachFromCurrentScope(
    tvm.getParamsFromCacheByName(paramNames),
  );

  const embed = tvm.detachFromCurrentScope(vm.getFunction("embed"));

  const createKvCache = vm.getFunction("create_tir_paged_kv_cache");

  const clearKvCaches = tvm.detachFromCurrentScope(
    tvm.getGlobalFunc("vm.builtin.paged_attention_kv_cache_clear"),
  );

  const KVCacheAddSequence = tvm.detachFromCurrentScope(
    tvm.getGlobalFunc("vm.builtin.paged_attention_kv_cache_add_sequence"),
  );

  // const KVCacheRemoveSequence = tvm.detachFromCurrentScope(
  //   tvm.getGlobalFunc('vm.builtin.paged_attention_kv_cache_remove_sequence')
  // )

  const KVCacheBeginForward = tvm.detachFromCurrentScope(
    tvm.getGlobalFunc("vm.builtin.paged_attention_kv_cache_begin_forward"),
  );

  const KVCacheEndForward = tvm.detachFromCurrentScope(
    tvm.getGlobalFunc("vm.builtin.paged_attention_kv_cache_end_forward"),
  );

  const defaultPageSize = 16;
  const defaultMaxNumSequence = 1;

  let maxWindowLength;
  {
    if ("contextWindowSize" in spec) {
      maxWindowLength = spec.contextWindowSize;
    } else if (
      "context_window_size" in metadata &&
      metadata.context_window_size != -1
    ) {
      maxWindowLength = metadata.context_window_size;
    } else if (
      "max_window_size" in metadata &&
      metadata.max_window_size != -1
    ) {
      maxWindowLength = metadata.max_window_size;
    } else {
      throw new Error(
        "Missing max window length, need either max_window_size or context_window_size",
      );
    }
  }

  let prefillChunkSize = -1;
  {
    if ("prefill_chunk_size" in metadata && metadata.prefill_chunk_size > 0) {
      prefillChunkSize = metadata.prefill_chunk_size;
    }
  }

  const kvCache = tvm.detachFromCurrentScope(
    createKvCache(
      tvm.makeShapeTuple([defaultMaxNumSequence]), // max_num_sequence
      tvm.makeShapeTuple([maxWindowLength]), // max_total_sequence_length
      tvm.makeShapeTuple([prefillChunkSize]), // prefill_chunk_size
      tvm.makeShapeTuple([defaultPageSize]), // page_size, hard coded for now
    ),
  );

  let filledKvCacheLength = 0;

  KVCacheAddSequence(kvCache, new Scalar(0, "int64"));

  tvm.endScope();

  if (targetDevice === TargetDevice.GPU) {
    updateReport({ loadGPUShaders: "waiting" });
    isLoadingGpuShaders = true;

    await tvm.asyncLoadWebGPUPipelines(vm.getInternalModule());
    updateReport({ loadGPUShaders: "done" });
  }

  const tokenize = (
    text: string,
    prefix: number[] = [],
    postfix: number[] = [],
  ) => {
    // TODO figure out if we've exceeded max window size and handle
    const encodedText = tokenizer.encode(text);

    return [...prefix, ...encodedText, ...postfix];
  };

  const logitsOnCpuCopyFromAndDispose = (() => {
    let logitsOnCpu: NDArray | undefined;

    return async (ndarray: DeviceNDArray): Promise<CpuNDArray> => {
      // WTB linear types
      const logits = ndarray.data;

      tvm.beginScope();

      if (logitsOnCpu === undefined) {
        logitsOnCpu = tvm.detachFromCurrentScope(
          tvm.empty(logits.shape, logits.dtype, tvm.cpu()),
        );
      } else {
        if (logits.shape[0] != logitsOnCpu.shape[0]) {
          throw Error("Logits changed shape");
        }
      }

      logitsOnCpu.copyFrom(logits);
      logits.dispose();

      tvm.endScope();

      await device?.sync();

      return {
        data: logitsOnCpu,
        host: "cpu",
      };
    };
  })();

  const sampleTokenFromLogits = (
    ndarray: CpuNDArray,
    temperature: number,
    top_p: number,
  ) => {
    return tvm.sampleTopPFromLogits(ndarray.data, temperature, top_p);
  };

  const prefillStep = (text: string): DeviceNDArray => {
    const tokens = tokenize(text);
    tvm.beginScope();

    const inputNdArray = tvm.empty([1, tokens.length], "int32", device);
    inputNdArray.copyFrom(tokens);

    const seqLen = inputNdArray.shape[1];

    const seqIdsTuple = tvm.makeShapeTuple([0]);
    const inputLenShape = tvm.makeShapeTuple([seqLen]);

    KVCacheBeginForward(kvCache, seqIdsTuple, inputLenShape);
    const embed_ = embed(inputNdArray, params);
    const retValue = prefill(embed_, kvCache, params);
    KVCacheEndForward(kvCache);

    const logits = tvm.detachFromCurrentScope(retValue.get(0));
    tvm.endScope();

    filledKvCacheLength += tokens.length;

    return {
      host: "dev",
      data: logits,
    };
  };

  const decodeStep = (lastToken: number): DeviceNDArray => {
    tvm.beginScope();

    const inputNdArray = tvm.empty([1, 1], "int32", device);
    inputNdArray.copyFrom([lastToken]);

    const seqIdsTuple = tvm.makeShapeTuple([0]);
    const appendLength = tvm.makeShapeTuple([1]);

    KVCacheBeginForward(kvCache, seqIdsTuple, appendLength);
    const embed_ = embed(inputNdArray, params);
    const retValue = decode(embed_, kvCache, params);
    KVCacheEndForward(kvCache);

    const logits = tvm.detachFromCurrentScope(retValue.get(0));

    tvm.endScope();

    filledKvCacheLength += 1;

    return {
      data: logits,
      host: "dev",
    };
  };

  let modelState: ModelState = ModelState.Waiting;

  let totalTokenCount = 0;

  const unfill = () => {
    clearKvCaches(kvCache);
    filledKvCacheLength = 0;
    KVCacheAddSequence(kvCache, new Scalar(0, "int64"));
  };

  const acceptTokenInference = (
    tokenizer: Tokenizer,
    stops: string[],
    prompt: string,
    stream?: GenerationStreamHandler,
  ) => {
    let tokens: number[] = [];
    let completion = "";

    return {
      accept: (token: number) => {
        tokens.push(token);
        totalTokenCount += 1;

        if (stopTokens.includes(token)) {
          return false;
        }

        const updatedText = tokenizer.decode(new Int32Array(tokens));
        const tokenDecodedText = updatedText.slice(completion.length);

        const stopIdx = getStopIndex(updatedText, tokenDecodedText, stops);

        if (stopIdx !== -1) {
          const acceptedCompleteText = updatedText.slice(0, stopIdx);

          stream?.({
            content: acceptedCompleteText.slice(completion.length),
            type: "gen",
            prompt,
          });

          completion = acceptedCompleteText;

          return false;
        }

        stream?.({
          content: tokenDecodedText,
          type: "gen",
          prompt,
        });

        completion = updatedText;
        return true;
      },
      get completion() {
        return completion;
      },
      get tokens() {
        return tokens;
      },
    };
  };

  const generate: LoadedModel["generate"] = async (
    { prompt, priorCompletion, stops, system, preprompt },
    options?: GenerateOptions,
  ): Promise<string> => {
    modelState = ModelState.Running as ModelState;

    const options_ = {
      temperature: 1.0,
      top_p: 0.95,
      maxTokens: 400,
      ...options,
    };

    const accepted = acceptTokenInference(
      tokenizer,
      stops,
      prompt,
      options_?.stream,
    );

    const buildSampler = options_?.sampler;
    const sample = buildSampler
      ? buildSampler(
          priorCompletion,
          stops,
          options_.temperature,
          options_.top_p,
        )
      : (logits: CpuNDArray) =>
          sampleTokenFromLogits(logits, options_.temperature, options_.top_p);

    const prefillText = `<<sys>>${system ?? "You are a helpful assistant"}<</sys>>\n\n[INST]${preprompt ? ` ${preprompt}` : ""} ${prompt} [/INST] ${priorCompletion}`;
    console.info("[generate:start]", prompt, { ...options_, prefillText });

    if (filledKvCacheLength > 0) {
      unfill();
    }

    const stopPrefillTimer = perf.timer("prefill");
    const nextToken = sample(
      await logitsOnCpuCopyFromAndDispose(prefillStep(prefillText)),
      accepted.tokens,
      accepted.completion,
    );
    stopPrefillTimer();

    if (nextToken === undefined) {
      throw Error("Prefilled with no sampled next token");
    }

    const continueSampling = accepted.accept(nextToken); // will be false if our first char was a stop

    if (continueSampling) {
      while (
        !(modelState === ModelState.Cancelling) &&
        accepted.tokens.length < options_.maxTokens
      ) {
        const tokens = accepted.tokens;

        const stopDecodeTimer = perf.timer("decode");
        const nextToken = sample(
          await logitsOnCpuCopyFromAndDispose(
            decodeStep(tokens[tokens.length - 1]),
          ),
          tokens,
          accepted.completion,
        );
        stopDecodeTimer();

        if (!accepted.accept(nextToken)) {
          break;
        }
      }
    }

    // TODO eos token

    if (modelState === ModelState.Cancelling) {
      modelState = ModelState.Waiting;
      unfill();
      throw Error("Model cancelled");
    }

    modelState = ModelState.Waiting;

    if (options_?.validate) {
      if (
        options_.validate.check &&
        !options_.validate.check(accepted.completion)
      ) {
        if (options_.validate.retries && options_.validate.retries > 0) {
          options_?.stream?.({
            type: "ungen",
            tokenCount: accepted.tokens.length,
            content: accepted.completion,
          });

          console.info("[validation-failed]", accepted.completion);

          return await generate(
            { prompt, priorCompletion, stops },
            {
              ...options_,
              validate: {
                ...options_.validate,
                retries: options_.validate.retries - 1,
              },
            },
          );
        } else {
          console.warn("Expression failed validation but ran out of retries", {
            completion: accepted.completion,
            retries: options_.validate.retries ?? 0,
          });
        }
      }

      if (options_.validate.transform) {
        // We transform even if validation fails due to exhausting retries. Should we only transform if validate succeeds?
        options_?.stream?.({
          type: "ungen",
          tokenCount: accepted.tokens.length,
          content: accepted.completion,
        });

        const transformed = options_.validate.transform(accepted.completion);

        options_?.stream?.({
          type: "gen",
          content: transformed,
          prompt,
        });

        return transformed;
      }
    }

    perf.summarize();

    console.info("[generate:done]", prompt, {
      acceptedCount: accepted.tokens.length,
      completion: accepted.completion,
    });

    return accepted.completion;
  };

  const bias = buildBias({ tvm, tokenizer, sample: sampleTokenFromLogits });

  updateReport({ ready: true });

  return {
    generate: async (params, options?) => {
      try {
        return await generate(params, options);
      } catch (e) {
        unfill();
        modelState = ModelState.Waiting;
        throw e;
      }
    },
    bias,
    cancel: async () => {
      if (modelState === ModelState.Running) {
        modelState = ModelState.Cancelling;
      }

      return new Promise<void>((resolve) =>
        setInterval(() => {
          if (modelState === ModelState.Waiting) {
            resolve();
          }
        }, 16),
      );
    },
    get totalTokenCount() {
      return totalTokenCount;
    },
    encode: (x: string) => Array.from(tokenizer.encode(x)),
    decode: (xs: number[]) => tokenizer.decode(new Int32Array(xs)),
  };
};
