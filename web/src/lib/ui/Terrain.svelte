<script>
  // The Live instrument's screen: a contour map whose hills are the genres. The listener is the dot. Dragging it (or the
  // arrow keys) steers the music; Drift makes it wander on its own. The contours move with the music, and pulse on the
  // beat for genres with a tempo.
  import { onMount } from "svelte";
  import { blendAt, contour, height, regions } from "./terrain.js";

  /**
   * @type {{
   *   genres: any[], x: number, y: number, drift: number, landmarks: {x: number, y: number, word: string}[],
   *   loudness: () => number, bpm: number | null, playing: boolean,
   *   onmove: (p: {x: number, y: number, mix: any, byHand: boolean, settled: boolean}) => void
   * }}
   */
  let { genres, x = 0.36, y = 0.5, drift = 3, landmarks = [], loudness = () => 0, bpm = null, playing = false, onmove } = $props();

  let canvas;
  const rs = $derived(regions(genres));
  const reduce = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  let dot = { x, y, vx: 0, vy: 0 };
  let trail = [];
  let dragging = false;
  let mixText = $state("");

  // Keep in step when the page moves the dot (a legend click, a resumed session).
  $effect(() => {
    if (!dragging && (Math.abs(dot.x - x) > 1e-3 || Math.abs(dot.y - y) > 1e-3)) {
      dot.x = x;
      dot.y = y;
    }
  });

  function report(byHand, settled) {
    const mix = blendAt(rs, dot.x, dot.y);
    mixText = mix.blend ? `${mix.genre.name} ${Math.round((1 - mix.blend.share) * 100)} · ${mix.blend.genre.name} ${Math.round(mix.blend.share * 100)}` : mix.genre.name;
    onmove?.({ x: dot.x, y: dot.y, mix, byHand, settled });
  }

  const clamp = (v) => Math.min(0.97, Math.max(0.03, v));
  function toLocal(e) {
    const r = canvas.getBoundingClientRect();
    return { x: clamp((e.clientX - r.left) / r.width), y: clamp((e.clientY - r.top) / r.height) };
  }
  function down(e) {
    canvas.setPointerCapture(e.pointerId);
    dragging = true;
    Object.assign(dot, toLocal(e), { vx: 0, vy: 0 });
    report(true, false);
  }
  function move(e) {
    if (!dragging) return;
    Object.assign(dot, toLocal(e));
    report(true, false);
  }
  function up() {
    if (!dragging) return;
    dragging = false;
    report(true, true);
  }
  let keyTimer;
  function key(e) {
    const d = e.shiftKey ? 0.08 : 0.03;
    const m = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, -d], ArrowDown: [0, d] }[e.key];
    if (!m) return;
    e.preventDefault();
    dot.x = clamp(dot.x + m[0]);
    dot.y = clamp(dot.y + m[1]);
    report(true, false);
    clearTimeout(keyTimer);
    keyTimer = setTimeout(() => report(true, true), 400);
  }

  onMount(() => {
    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1;
    const resize = () => {
      dpr = Math.min(2, devicePixelRatio || 1);
      const r = canvas.getBoundingClientRect();
      W = canvas.width = Math.round(r.width * dpr);
      H = canvas.height = Math.round(r.height * dpr);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    const GX = 72, GY = 44;
    const field = new Float32Array((GX + 1) * (GY + 1));
    let level = 0;
    let last = performance.now();
    let lastReport = 0;
    let raf;

    function frame(now) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const t = reduce ? 0 : now / 1000;

      // Drift: the dot wanders on its own, faster as the knob goes up. It reports gently every few seconds.
      if (!dragging && drift > 0 && !reduce) {
        const k = drift / 10;
        dot.vx += (Math.sin(t * 0.37 + 1.3) * 0.5 + (Math.random() - 0.5)) * k * dt * 0.25;
        dot.vy += (Math.cos(t * 0.29) * 0.5 + (Math.random() - 0.5)) * k * dt * 0.25;
        dot.vx *= 0.97;
        dot.vy *= 0.97;
        dot.x += dot.vx * dt;
        dot.y += dot.vy * dt;
        if (dot.x < 0.05 || dot.x > 0.95) dot.vx *= -1;
        if (dot.y < 0.05 || dot.y > 0.95) dot.vy *= -1;
        dot.x = clamp(dot.x);
        dot.y = clamp(dot.y);
        if (now - lastReport > 2500) {
          lastReport = now;
          report(false, true);
        }
      }
      const tail = trail[trail.length - 1];
      if (!tail || Math.hypot(tail.x - dot.x, tail.y - dot.y) > 0.006) {
        trail.push({ x: dot.x, y: dot.y });
        if (trail.length > 300) trail.shift();
      }

      // The music moves the terrain.
      level += (Math.min(1, loudness() * 4) - level) * 0.12;
      const beat = playing && bpm && !reduce ? Math.max(0, Math.cos(t * Math.PI * 2 * (bpm / 60))) ** 10 : 0;

      ctx.fillStyle = "#0b0a16";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(169,147,240,0.06)";
      ctx.lineWidth = dpr;
      ctx.beginPath();
      for (let i = 1; i < 12; i++) {
        ctx.moveTo((i / 12) * W, 0);
        ctx.lineTo((i / 12) * W, H);
      }
      for (let j = 1; j < 8; j++) {
        ctx.moveTo(0, (j / 8) * H);
        ctx.lineTo(W, (j / 8) * H);
      }
      ctx.stroke();

      for (let j = 0; j <= GY; j++) for (let i = 0; i <= GX; i++) field[j * (GX + 1) + i] = height(rs, landmarks, i / GX, j / GY, t);
      const levels = 13;
      for (let L = 1; L <= levels; L++) {
        const k = L / levels;
        const iso = k * 1.05 + 0.03 * (beat + level * 0.5);
        const c = [Math.round(91 + 164 * k), Math.round(74 + 137 * k), Math.round(158 + 3 * k)];
        ctx.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.28 + 0.55 * k})`;
        ctx.lineWidth = (1 + k * 1.2 + (beat * 0.6 + level * 0.6) * k) * dpr;
        ctx.beginPath();
        for (const [x1, y1, x2, y2] of contour(field, GX, GY, iso)) {
          ctx.moveTo(x1 * W, y1 * H);
          ctx.lineTo(x2 * W, y2 * H);
        }
        ctx.stroke();
      }

      // Labels sit on a dark halo so they read over the contour lines.
      ctx.textAlign = "center";
      ctx.lineJoin = "round";
      const label = (text, x0, y0, fill) => {
        ctx.strokeStyle = "rgba(11,10,22,0.9)";
        ctx.lineWidth = 5 * dpr;
        ctx.strokeText(text, x0, y0);
        ctx.fillStyle = fill;
        ctx.fillText(text, x0, y0);
      };
      ctx.font = `600 ${12 * dpr}px "Sora Variable", sans-serif`;
      if (ctx.letterSpacing !== undefined) ctx.letterSpacing = `${1.5 * dpr}px`;
      for (const r of rs) label(r.genre.name.toUpperCase(), r.x * W, r.y * H - 22 * dpr, `rgba(${r.color.join(",")},0.95)`);
      if (ctx.letterSpacing !== undefined) ctx.letterSpacing = "0px";
      ctx.font = `500 ${12 * dpr}px "JetBrains Mono Variable", monospace`;
      for (const l of landmarks) label(`◆ ${l.word}`, l.x * W, l.y * H - 10 * dpr, "rgba(255,217,179,0.95)");

      ctx.strokeStyle = "rgba(255,211,161,0.4)";
      ctx.lineWidth = 2 * dpr;
      ctx.setLineDash([2 * dpr, 6 * dpr]);
      ctx.beginPath();
      trail.forEach((p, i) => (i ? ctx.lineTo(p.x * W, p.y * H) : ctx.moveTo(p.x * W, p.y * H)));
      ctx.stroke();
      ctx.setLineDash([]);

      const px = dot.x * W;
      const py = dot.y * H;
      const swell = 1 + level * 0.6 + beat * 0.35;
      const glow = ctx.createRadialGradient(px, py, 0, px, py, 44 * dpr * swell);
      glow.addColorStop(0, "rgba(255,211,161,0.45)");
      glow.addColorStop(1, "rgba(255,211,161,0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(px, py, 44 * dpr * swell, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,211,161,0.7)";
      ctx.lineWidth = 2.5 * dpr;
      ctx.beginPath();
      ctx.arc(px, py, 16 * dpr * swell, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "#ffd3a1";
      ctx.beginPath();
      ctx.arc(px, py, 8 * dpr, 0, Math.PI * 2);
      ctx.fill();

      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);
    report(false, true);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  });
</script>

<div class="screen">
  <canvas
    bind:this={canvas}
    tabindex="0"
    aria-label="Terrain: drag the dot between genres, or use the arrow keys (Shift for bigger steps)"
    onpointerdown={down}
    onpointermove={move}
    onpointerup={up}
    onpointercancel={up}
    onkeydown={key}
  ></canvas>
  <p class="mix" aria-live="polite">{mixText}</p>
</div>

<style>
  .screen { position: relative; border-radius: 16px; background: #0b0a16; padding: 0; overflow: hidden;
    box-shadow: inset 0 2px 14px rgba(0, 0, 0, 0.9), 0 0 0 1px #2d2650; }
  canvas { display: block; width: 100%; aspect-ratio: 16 / 10; touch-action: none; cursor: grab; }
  canvas:active { cursor: grabbing; }
  canvas:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .mix { position: absolute; left: 14px; bottom: 10px; margin: 0; pointer-events: none; font: 500 11px var(--mono);
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--apricot); text-shadow: 0 0 6px rgba(255, 180, 120, 0.5); }
</style>
