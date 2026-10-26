// The "Try it" page: send a question to the API and draw every step of the answer.
import { runQuery } from "./api.js";
import { bar, clear, h, kv } from "./dom.js";
import { isServerReady, llmMode } from "./status.js";
import { ms, number, percent, perThousand, usd } from "./format.js";

const EXAMPLES = [
  ["Easy", "What does CPU stand for?"],
  ["Medium", "Explain how a hash map works and why lookups are fast."],
  ["Hard", "Design a multi region database failover plan, compare three replication strategies, and list the risks and rollback steps."],
  ["Docs", "How do spot instances lower my GPU bill?"],
  ["Docs", "How does semantic caching avoid false hits?"],
  ["Cache", "What is the capital of France?"],
  ["Cache", "whats the capital of france"],
  ["Cache", "What is the capital of Spain?"],
];

const TIER_OF_LABEL = { simple: "small", moderate: "medium", complex: "large" };

const form = document.getElementById("query-form");
const input = document.getElementById("query-input");
const counter = document.getElementById("char-count");
const button = document.getElementById("run-button");
const hint = document.getElementById("form-hint");
const result = document.getElementById("result");

// ---- the form ---------------------------------------------------------

function buildExamples() {
  const chips = document.getElementById("example-chips");
  for (const [kind, text] of EXAMPLES) {
    chips.append(
      h("button", {
        class: "chip",
        type: "button",
        onclick: () => {
          input.value = text;
          updateCounter();
          input.focus();
        },
      }, h("span", { class: "kind" }, kind), text),
    );
  }
}

function updateCounter() {
  counter.textContent = `${input.value.length} / ${input.maxLength}`;
}

async function submit(event) {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) {
    hint.textContent = "Type a question first.";
    input.focus();
    return;
  }
  const priority = form.elements.priority.value;

  button.disabled = true;
  button.textContent = "Working";
  hint.textContent = isServerReady()
    ? ""
    : "The free server was asleep. The first question can take up to a minute.";

  try {
    render(await runQuery(query, priority));
    hint.textContent = "";
  } catch (error) {
    clear(result).append(h("p", { class: "error-box", role: "alert" }, error.message));
    hint.textContent = "";
  } finally {
    button.disabled = false;
    button.textContent = "Run question";
  }
}

// ---- drawing the result -------------------------------------------------

function card(title, ...children) {
  return h("section", { class: "card" }, h("h3", {}, title), ...children);
}

function render(data) {
  clear(result).append(
    answerCard(data),
    h("div", { class: "result-grid" },
      costCard(data.cost),
      routeCard(data.routing),
      cacheCard(data.cache),
      retrievalCard(data.retrieval),
      computeCard(data.compute),
      timingCard(data.timings),
    ),
  );
  result.firstChild.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function answerCard(data) {
  const badges = {
    live: `Live answer from ${data.routing.model_id}`,
    offline: "Offline answer quoted from the documents",
    cache: "Answer reused from the cache",
  };
  return h("section", { class: "card answer-card" },
    h("h2", { style: "margin-top:0" }, "Answer"),
    h("span", { class: `badge ${data.answer_mode === "cache" ? "" : `tier-${data.routing.tier}`}` }, badges[data.answer_mode]),
    h("p", { class: "answer-text" }, data.answer),
    data.warning && h("p", { class: "notice" }, data.warning),
    data.answer_mode === "offline" && llmMode() === "offline" &&
      h("p", { class: "muted small" }, "This server has no language model key. Set ANTHROPIC_API_KEY on the backend to switch on live answers."),
  );
}

function costCard(cost) {
  const worth = cost.saved_percent > 0;
  return card("Cost",
    h("p", { class: "saved-number" }, worth ? `${cost.saved_percent}% cheaper` : "No saving here"),
    h("p", { class: "muted small" },
      worth
        ? `than sending this question to ${cost.baseline_model}`
        : "This question needs the large model, so SmartStack uses it too."),
    h("div", { class: "cost-bars" },
      costRow("Always the large model", cost.baseline_usd, cost.baseline_usd, "baseline"),
      costRow("With SmartStack", cost.actual_usd, cost.baseline_usd, ""),
    ),
    kv([
      ["Tokens in", number(cost.input_tokens)],
      ["Tokens out", number(cost.output_tokens)],
    ]),
    h("p", { class: "muted small" }, "Prices shown for 1,000 questions like this one. Based on example prices."),
  );
}

function costRow(label, value, max, extraClass) {
  return h("div", { class: "cost-row" },
    h("div", { class: "label" }, h("span", {}, label), h("strong", {}, perThousand(value))),
    bar(max ? Math.max(value / max, value > 0 ? 0.02 : 0) : 0, extraClass),
  );
}

function routeCard(routing) {
  const order = ["simple", "moderate", "complex"];
  const bars = h("div", { class: "prob-list" });
  for (const label of order) {
    const p = routing.probabilities[label] ?? 0;
    bars.append(
      h("div", { class: "cost-row" },
        h("div", { class: "label" }, h("span", {}, label), h("span", {}, percent(p))),
        bar(p, `${TIER_OF_LABEL[label]}-bar`),
      ),
    );
  }
  return card("Route",
    h("p", {}, "Chosen model: ", h("span", { class: `badge tier-${routing.tier}` }, `${routing.tier}`)),
    bars,
    h("p", { class: "muted small" }, routing.reason),
    h("p", { class: "muted small" }, `Model id: ${routing.model_id}`),
  );
}

function cacheCard(cache) {
  if (!cache.hit) {
    return card("Cache",
      h("p", {}, h("strong", {}, "Miss.")),
      h("p", { class: "muted small" }, "No similar question was stored. This answer is now saved for an hour, so asking again is free."),
    );
  }
  return card("Cache",
    h("p", {}, h("strong", {}, "Hit.")),
    kv([["Letter similarity", cache.similarity.toFixed(2)]]),
    h("p", { class: "muted small" }, `Matched the earlier question: "${cache.matched_query}"`),
  );
}

function retrievalCard(retrieval) {
  if (!retrieval.used) {
    return card("Retrieve",
      h("p", {}, h("strong", {}, "Nothing sent.")),
      h("p", { class: "muted small" }, "No passage matched well enough, so no documents were added to the prompt. That saves tokens."),
    );
  }
  const list = h("div");
  for (const source of retrieval.sources) {
    list.append(
      h("div", { class: "source" },
        h("div", { class: "source-head" },
          h("strong", {}, source.title),
          h("span", { class: "muted small" }, `score ${source.score.toFixed(2)}`),
        ),
        h("p", {}, source.snippet),
      ),
    );
  }
  return card("Retrieve", list);
}

function computeCard(plan) {
  if (!plan) {
    return card("Plan", h("p", {}, h("strong", {}, "Not needed.")), h("p", { class: "muted small" }, "The cache answered, so nothing had to run."));
  }
  return card("Plan",
    kv([
      ["Provider", plan.provider],
      ["Region", plan.region],
      ["Machine", `${plan.gpu}, ${plan.pricing}`],
      ["Price per hour", `${usd(plan.price_per_hour)} (default ${usd(plan.baseline_price_per_hour)})`],
      ["Cheaper by", `${plan.saving_percent}%`],
      ["Latency", `${plan.latency_ms} ms`],
    ]),
    h("p", { class: "muted small" }, plan.note),
    h("p", { class: "muted small" }, "A recommendation based on example prices. Nothing was started."),
  );
}

function timingCard(timings) {
  return card("Time spent",
    kv([
      ["Route", ms(timings.route_ms)],
      ["Cache", ms(timings.cache_ms)],
      ["Retrieve", ms(timings.retrieve_ms)],
      ["Answer", ms(timings.generate_ms)],
      ["Total", ms(timings.total_ms)],
    ]),
  );
}

export function initTryPage() {
  buildExamples();
  updateCounter();
  input.addEventListener("input", updateCounter);
  form.addEventListener("submit", submit);
}
