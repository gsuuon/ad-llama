const parses = (json: string, assertx = (_: any) => true) => {
  try {
    return assertx(JSON.parse(json).x)
  } catch {}

  return false
}

export const json = {
  list: (x: string, pre = '') => parses(`{"x":[${pre}${x}]}`),
  str: (x: string, pre = '') => parses(`{"x":"${pre}${x}"}`),
  bool: (x: string, pre = '') => parses(`{"x":${pre}${x}}`, x => typeof(x) === 'boolean'),
  num: (x: string, pre = '') => parses(`{"x":${pre}${x}}`, x => typeof(x) === 'number') // TODO not sure if this is right
}
