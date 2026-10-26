// The dashboard page: totals for every question since the server started.
import { getStats } from "./api.js";
import { bar, clear, h } from "./dom.js";
import { number, usd } from "./format.js";
import { serverReady } from "./status.js";

const container = document.getElementById("dashboard");
let timer = null;

function tile(name, value, note) {
  return h("div", { class: "tile" },
    h("div", { class: "value" }, value),
    h("div", { class: "name" }, name),
    note && h("div", { class: "muted small" }, note),
  );
}

function render(stats) {
  clear(container);
  if (stats.requests === 0) {
    container.append(h("p", { class: "empty-state" }, "No questions yet. Ask one on the Try it page and the totals appear here."));
    return;
  }
  const hitRate = stats.cache_hits / stats.requests;
  container.append(
    h("div", { class: "tiles" },
      tile("Questions answered", number(stats.requests)),
      tile("Cache hits", number(stats.cache_hits), `${Math.round(hitRate * 100)}% of questions`),
      tile("Saved", `${stats.saved_percent}%`, `${usd(stats.saved_usd)} less than the large model`),
      tile("Moved up a tier", number(stats.escalations), "because the classifier was unsure"),
    ),
    h("div", { class: "card" },
      h("h2", {}, "Where the money went"),
      h("div", { class: "cost-bars" },
        costRow("Always the large model", stats.baseline_usd, stats.baseline_usd, "baseline"),
        costRow("With SmartStack", stats.actual_usd, stats.baseline_usd, ""),
      ),
    ),
    tierCard(stats),
  );
}

function costRow(label, value, max, extraClass) {
  return h("div", { class: "cost-row" },
    h("div", { class: "label" }, h("span", {}, label), h("strong", {}, usd(value))),
    bar(max ? value / max : 0, extraClass),
  );
}

function tierCard(stats) {
  const answered = Object.values(stats.by_tier).reduce((a, b) => a + b, 0);
  const card = h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "Which model answered"));
  if (answered === 0) {
    card.append(h("p", { class: "muted" }, "Every question so far was answered from the cache."));
    return card;
  }
  const rows = h("div", { class: "cost-bars" });
  for (const [tier, count] of Object.entries(stats.by_tier)) {
    rows.append(
      h("div", { class: "cost-row" },
        h("div", { class: "label" }, h("span", {}, `${tier[0].toUpperCase()}${tier.slice(1)} model`), h("strong", {}, number(count))),
        bar(count / answered, `${tier}-bar`),
      ),
    );
  }
  card.append(rows);
  return card;
}

async function refresh() {
  try {
    render(await getStats());
  } catch {
    clear(container).append(h("p", { class: "error-box", role: "alert" }, "Could not load the totals. The server may be asleep; try again in a moment."));
  }
}

export async function startDashboard() {
  await serverReady;
  await refresh();
  clearInterval(timer);
  timer = setInterval(refresh, 10000);
}

export function stopDashboard() {
  clearInterval(timer);
  timer = null;
}

document.getElementById("refresh-stats").addEventListener("click", refresh);
