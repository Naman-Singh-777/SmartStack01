// Watches whether the API is awake and shows it in the header.
// Free hosting puts idle servers to sleep, so the first visit may need a minute.
import { getHealth } from "./api.js";

const box = document.getElementById("api-status");
const text = document.getElementById("api-status-text");

function show(state, message) {
  box.dataset.state = state;
  text.textContent = message;
}

let ready = false;
let info = null;

/** Resolves with the health info as soon as the API answers. */
export const serverReady = new Promise((resolve) => {
  connect(resolve);
});

export const isServerReady = () => ready;
export const llmMode = () => info?.llm_mode ?? "offline";

async function connect(resolve) {
  const maxTries = 24;
  for (let attempt = 1; attempt <= maxTries; attempt++) {
    try {
      info = await getHealth();
      ready = true;
      show("online", `API online, ${info.llm_mode} answers`);
      resolve(info);
      return;
    } catch {
      show("waking", attempt === 1 ? "Connecting to the API" : "Waking the free server, up to a minute");
      await new Promise((wait) => setTimeout(wait, 4000));
    }
  }
  show("offline", "API unreachable, reload to retry");
}
