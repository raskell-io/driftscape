import adapter from "@sveltejs/adapter-static";
import { sveltekit } from "@sveltejs/kit/vite";
import { defaultClientConditions, defineConfig } from "vite";

// The Hugging Face hosts the model downloads from (the repo, then its CDN redirects).
const MODEL_HOSTS = ["https://huggingface.co", "https://*.huggingface.co", "https://*.hf.co"];

export default defineConfig({
  plugins: [
    sveltekit({
      // Static output only: every route is prerendered, so the app is hosted as files with no server.
      adapter: adapter({ fallback: "404.html" }),
      // A strict Content Security Policy as a <meta> tag, with hashes for SvelteKit's inline bootstrap script.
      // 'wasm-unsafe-eval' is what WebAssembly needs; connect-src adds the model hosts and nothing else.
      csp: {
        mode: "hash",
        directives: {
          "default-src": ["none"],
          "script-src": ["self", "wasm-unsafe-eval"],
          "worker-src": ["self"],
          "style-src": ["self", "unsafe-hashes", "sha256-S8qMpvofolR8Mpjy4kQvEm7m1q8clzU4dfDH0AmvZjo="],
          "font-src": ["self"],
          "img-src": ["self", "data:"],
          "media-src": ["self", "blob:"],
          "connect-src": ["self", ...MODEL_HOSTS],
          "manifest-src": ["self"],
          "base-uri": ["none"],
          "form-action": ["self"],
        },
      },
    }),
  ],
  // Use onnxruntime-web's build that loads its runtime from static/ort/ at run time. The default build points Vite at
  // the 28 MB wasm, which would be copied into the output and is too large for Cloudflare Pages.
  resolve: { conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions] },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
});
