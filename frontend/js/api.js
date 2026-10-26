// Every call to the backend goes through this file.
import { API_BASE } from "./config.js";

export class ApiError extends Error {}

async function request(path, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(API_BASE + path, { ...options, signal: controller.signal });
    if (!response.ok) throw new ApiError(messageForStatus(response.status));
    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error.name === "AbortError") {
      throw new ApiError("The server took too long to answer. Please try again in a moment.");
    }
    throw new ApiError("Could not reach the API. Check your connection and try again.");
  } finally {
    clearTimeout(timer);
  }
}

function messageForStatus(status) {
  if (status === 422) return "Please type a question of up to 1000 characters.";
  if (status === 429) return "That was a lot of questions. Please wait a minute and try again.";
  return `The server had a problem (error ${status}). Please try again.`;
}

export const getHealth = () => request("/api/health", {}, 8000);
export const getStats = () => request("/api/stats");

let modelsPromise = null;
/** The model ladder never changes while the page is open, so ask only once. */
export function getModels() {
  modelsPromise ??= request("/api/models").catch((error) => {
    modelsPromise = null; // allow a retry later
    throw error;
  });
  return modelsPromise;
}

// A free server that has been idle can take up to a minute to wake, hence the long timeout.
export const runQuery = (query, priority) =>
  request(
    "/api/query",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, priority }),
    },
    75000,
  );
