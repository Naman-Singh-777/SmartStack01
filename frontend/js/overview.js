// The "three sizes of model" cards on the overview page, and the facts on "How it works".
import { getModels } from "./api.js";
import { h, clear } from "./dom.js";
import { percent } from "./format.js";
import { serverReady } from "./status.js";

export async function loadOverview() {
  await serverReady;
  const grid = document.getElementById("tier-grid");
  const facts = document.getElementById("classifier-facts");
  try {
    const models = await getModels();
    clear(grid);
    for (const tier of models.tiers) {
      grid.append(
        h("article", { class: `tier-card ${tier.key}-tier` },
          h("h3", {}, tier.label),
          h("p", { class: "model" }, tier.model_id),
          h("p", { class: "price" }, `$${tier.input_per_million}`, h("small", {}, " per million tokens in")),
          h("p", { class: "muted small" }, `$${tier.output_per_million} per million tokens out`),
        ),
      );
    }
    facts.textContent =
      `On questions it has not seen, the classifier is right about ${percent(models.classifier_accuracy)} ` +
      `of the time (5-fold cross-validation on ${models.training_examples} examples).`;
  } catch {
    grid.textContent = "Prices could not be loaded. Reload the page to try again.";
    facts.textContent = "Accuracy could not be loaded.";
  }
}
