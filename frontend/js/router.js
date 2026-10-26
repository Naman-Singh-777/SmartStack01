// Shows one page at a time, based on the part of the address after "#".
// "#/try" shows the section marked data-page="/try". No page reloads, no server setup.

const pages = [...document.querySelectorAll("[data-page]")];
const links = [...document.querySelectorAll("[data-route]")];
const main = document.getElementById("main");
const hooks = {}; // route -> { enter, leave }

export function onPage(route, enter, leave) {
  hooks[route] = { enter, leave };
}

let current = null;

function currentRoute() {
  const route = location.hash.replace(/^#/, "") || "/";
  return pages.some((page) => page.dataset.page === route) ? route : "/";
}

function show() {
  const route = currentRoute();
  if (route === current) return;
  hooks[current]?.leave?.();

  for (const page of pages) page.hidden = page.dataset.page !== route;
  for (const link of links) {
    if (link.dataset.route === route) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }

  current = route;
  document.title = `${titleFor(route)} | SmartStack`;
  window.scrollTo(0, 0);
  main.focus({ preventScroll: true }); // lets keyboard and screen reader users start at the new page
  hooks[route]?.enter?.();
}

function titleFor(route) {
  return { "/": "Overview", "/try": "Try it", "/how": "How it works", "/dashboard": "Dashboard" }[route];
}

export function startRouter() {
  window.addEventListener("hashchange", show);
  show();
}
