// Entry point: wires the pages together.
import { startDashboard, stopDashboard } from "./dashboard.js";
import { loadOverview } from "./overview.js";
import { onPage, startRouter } from "./router.js";
import "./status.js"; // starts checking whether the API is awake
import { initTryPage } from "./try.js";

initTryPage();

onPage("/", loadOverview);
onPage("/how", loadOverview);
onPage("/dashboard", startDashboard, stopDashboard);

startRouter();
