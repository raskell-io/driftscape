/// <reference types="@sveltejs/kit" />
// Offline support and safe updates.
//
// - The app's own files (the hashed build, static/ including the ONNX runtime and core wasm, and the pages) are cached
//   per deploy, under a cache named after the deploy's version, and served cache-first. With the model in Cache
//   Storage too (the voice worker's "driftscape-models-v1", never touched here), Driftscape works fully offline.
// - A new deploy installs in the background but does not take over a page that is playing: it waits until the page
//   asks for it ("skip-waiting", sent when the listener chooses to reload) or until every tab is closed. Only then are
//   older caches removed, so a running session never loses a file it still needs.
// - Pages are network-first with the cached copy as the offline fallback; /api/ (the native engine behind
//   `driftscape serve`) is never cached.
import { assets, immutable, prerendered } from "$app/manifest";
import { version } from "$app/env";

const CACHE = `app-${version}`;
const ASSETS = new Set([...immutable, ...assets, ...prerendered].map(({ path }) => (path.startsWith("/") ? path : "/" + path)));
// The three ONNX runtimes (4–7 MB each) are not precached: a browser only ever uses one. It is cached when first used.
const RUNTIME = (path) => path.startsWith("/ort/");
const PRECACHE = [...ASSETS].filter((p) => !RUNTIME(p));

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)));
});

self.addEventListener("message", (event) => {
  if (event.data?.type === "skip-waiting") self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("app-") && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api/")) return;
  if (ASSETS.has(url.pathname)) {
    event.respondWith(
      (async () => {
        const hit = await caches.match(url.pathname);
        if (hit) return hit;
        const res = await fetch(event.request);
        if (res.ok && RUNTIME(url.pathname)) (await caches.open(CACHE)).put(url.pathname, res.clone());
        return res;
      })(),
    );
    return;
  }
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);
      try {
        const res = await fetch(event.request);
        if (res.ok && event.request.mode === "navigate") cache.put(event.request, res.clone());
        return res;
      } catch {
        return (await cache.match(event.request)) || (await cache.match("/")) || Response.error();
      }
    })(),
  );
});
