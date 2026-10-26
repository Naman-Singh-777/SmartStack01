// Where the API lives. On your own computer it is the local server.
// On GitHub Pages it is the deployed backend on Render.
const onThisComputer = ["localhost", "127.0.0.1"].includes(location.hostname);

export const API_BASE = onThisComputer
  ? "http://localhost:8000"
  : "https://smartstack01-api.onrender.com";
