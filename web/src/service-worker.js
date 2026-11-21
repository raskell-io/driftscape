/// <reference types="@sveltejs/kit" />
// Offline support with SvelteKit's service worker. The app's own files are cached per deploy; the model (about
// 570 MB) is cached separately by the voice worker, in its own Cache Storage bucket, and is never touched here.
import { assets, immutable, prerendered } from "$app/manifest";
import { version } from "$app/env";

const CACHE = `app-${version}`;
const ASSETS = new Set([...immutable, ...assets, ...prerendered].map(({ path }) => (path.startsWith("/") ? path : "/" + path)));

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll([...ASSETS])).then(() => self.skipWaiting()));
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
    event.respondWith(caches.match(url.pathname).then((hit) => hit || fetch(event.request)));
    return;
  }
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);
      try {
        const res = await fetch(event.request);
        if (res.ok) cache.put(event.request, res.clone());
        return res;
      } catch {
        return (await cache.match(event.request)) || (await cache.match("/")) || Response.error();
      }
    })(),
  );
});
