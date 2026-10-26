// Tiny helpers for building page elements without writing HTML strings.
// Text goes in as text, never as HTML, so nothing from the API can inject markup.

/**
 * h("p", { class: "muted" }, "Hello ", h("strong", {}, "world"))
 * creates <p class="muted">Hello <strong>world</strong></p>
 */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === false || value == null) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2).toLowerCase(), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Remove everything inside an element. */
export function clear(node) {
  node.replaceChildren();
  return node;
}

/** A definition list of label/value pairs: kv([["Region", "us-east-1"]]) */
export function kv(pairs) {
  const list = h("dl", { class: "kv" });
  for (const [label, value] of pairs) {
    list.append(h("dt", {}, label), h("dd", {}, value));
  }
  return list;
}

/** A horizontal bar. `fraction` is between 0 and 1. */
export function bar(fraction, extraClass = "") {
  const width = Math.max(0, Math.min(1, fraction)) * 100;
  const fill = h("span");
  fill.style.width = `${width}%`;
  return h("div", { class: `bar ${extraClass}`.trim(), role: "presentation" }, fill);
}
