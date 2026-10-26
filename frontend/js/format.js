// Turn raw numbers into text people can read.

export function usd(value) {
  if (value === 0) return "$0";
  if (value >= 1) return `$${value.toFixed(2)}`;
  if (value >= 0.01) return `$${value.toFixed(3)}`;
  return `$${value.toFixed(6)}`;
}

/** One request costs a tiny fraction of a cent, so show the price of 1,000 requests instead. */
export function perThousand(value) {
  return usd(value * 1000);
}

export const percent = (fraction) => `${Math.round(fraction * 100)}%`;
export const ms = (value) => `${value.toFixed(value < 10 ? 1 : 0)} ms`;
export const number = (value) => value.toLocaleString("en-US");
