<script>
  // The Radio instrument: stations sit on a frequency band. Spin the dial; between stations there is soft static, and
  // resting on a station for a moment tunes in (onlock). A station you are near pulls the needle gently into place.
  import { onMount } from "svelte";

  /** @type {{ stations: any[], genres: any[], tunedId: string | null, onair: boolean, onlock: (id: string) => void }} */
  let { stations, genres, tunedId = null, onair = false, onlock } = $props();

  const LOCK = 1.2; // within this much of a station counts as tuned
  const HOLD_MS = 700;
  const freqOf = (i, n) => (n <= 1 ? 50 : 10 + (i * 80) / (n - 1));
  const placed = $derived(stations.map((s, i) => ({ ...s, f: freqOf(i, stations.length) })));

  let tune = $state(10);
  let angle = 0;
  let canvas, dimple;
  let touching = false;
  let lastTouch = 0;
  let lockedAt = 0;
  let lockedId = null;

  const nearest = $derived(placed.reduce((b, s) => (!b || Math.abs(s.f - tune) < Math.abs(b.f - tune) ? s : b), null));
  const locked = $derived(nearest && Math.abs(nearest.f - tune) < LOCK);
  const shown = $derived(nearest && Math.abs(nearest.f - tune) < 2.5 ? nearest : null);
  const genreName = (id) => genres.find((g) => g.id === id)?.name ?? "";

  // Start on the station that's playing, if any.
  $effect(() => {
    const s = placed.find((p) => p.id === tunedId);
    if (s && !touching && Date.now() - lastTouch > 1500) tune = s.f;
  });

  // ---- static ----
  let audio = null;
  function staticOn() {
    if (audio) return;
    const ctx = new AudioContext();
    const buf = new AudioBuffer({ length: ctx.sampleRate * 2, sampleRate: ctx.sampleRate });
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    const src = new AudioBufferSourceNode(ctx, { buffer: buf, loop: true });
    const band = new BiquadFilterNode(ctx, { type: "bandpass", frequency: 1800, Q: 0.7 });
    const gain = new GainNode(ctx, { gain: 0 });
    src.connect(band).connect(gain).connect(ctx.destination);
    src.start();
    audio = { ctx, gain };
  }
  function setStatic(amount) {
    if (audio) audio.gain.gain.setTargetAtTime(amount * 0.05, audio.ctx.currentTime, 0.08);
  }

  // ---- input ----
  let prevAngle = null;
  const angleOf = (e, el) => {
    const r = el.getBoundingClientRect();
    return Math.atan2(e.clientY - (r.top + r.height / 2), e.clientX - (r.left + r.width / 2));
  };
  function down(e) {
    e.currentTarget.setPointerCapture(e.pointerId);
    prevAngle = angleOf(e, e.currentTarget);
    touching = true;
    staticOn();
  }
  function move(e) {
    if (prevAngle === null) return;
    const a = angleOf(e, e.currentTarget);
    let d = a - prevAngle;
    if (d > Math.PI) d -= 2 * Math.PI;
    if (d < -Math.PI) d += 2 * Math.PI;
    prevAngle = a;
    angle += d;
    tune = Math.min(100, Math.max(0, tune + d * 6));
    lastTouch = Date.now();
  }
  function up() {
    prevAngle = null;
    touching = false;
    lastTouch = Date.now();
  }
  function key(e) {
    const step = e.shiftKey ? 5 : 1;
    if (["ArrowRight", "ArrowUp"].includes(e.key)) tune = Math.min(100, tune + step);
    else if (["ArrowLeft", "ArrowDown"].includes(e.key)) tune = Math.max(0, tune - step);
    else if (e.key === "Enter" && locked) return onlock?.(nearest.id);
    else return;
    e.preventDefault();
    angle += step * 0.15 * (["ArrowRight", "ArrowUp"].includes(e.key) ? 1 : -1);
    lastTouch = Date.now();
    staticOn();
  }

  onMount(() => {
    const ctx = canvas.getContext("2d");
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, dpr = 1, raf;
    const resize = () => {
      dpr = Math.min(2, devicePixelRatio || 1);
      const r = canvas.getBoundingClientRect();
      W = canvas.width = Math.round(r.width * dpr);
      H = canvas.height = Math.round(r.height * dpr);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    function frame(now) {
      const t = reduce ? 0 : now / 1000;
      const recent = touching || Date.now() - lastTouch < 1200;
      // A station you are near pulls the needle into place once you let go.
      if (!touching && nearest && Math.abs(nearest.f - tune) < 3) tune += (nearest.f - tune) * 0.06;
      const off = nearest ? Math.min(1, Math.abs(nearest.f - tune) / 5) : 1;
      setStatic(recent ? off : 0);

      // Tune in after resting on a station.
      if (locked && recent) {
        if (lockedId !== nearest.id) {
          lockedId = nearest.id;
          lockedAt = now;
        } else if (now - lockedAt > HOLD_MS && nearest.id !== tunedId) {
          onlock?.(nearest.id);
          lockedAt = Infinity;
        }
      } else if (!locked) {
        lockedId = null;
      }

      const x = (f) => (0.04 + (f / 100) * 0.92) * W;
      ctx.fillStyle = "#0b0a16";
      ctx.fillRect(0, 0, W, H);
      for (let i = 0; i <= 100; i++) {
        const major = i % 10 === 0;
        ctx.strokeStyle = major ? "rgba(233,220,207,0.6)" : "rgba(233,220,207,0.18)";
        ctx.lineWidth = (major ? 1.5 : 1) * dpr;
        ctx.beginPath();
        ctx.moveTo(x(i), H * 0.2);
        ctx.lineTo(x(i), H * (major ? 0.36 : 0.3));
        ctx.stroke();
        if (major) {
          ctx.fillStyle = "rgba(233,220,207,0.5)";
          ctx.font = `500 ${11 * dpr}px "JetBrains Mono Variable", monospace`;
          ctx.textAlign = "center";
          ctx.fillText((86 + i * 0.2).toFixed(0), x(i), H * 0.14);
        }
      }
      for (const s of placed) {
        const near = Math.max(0, 1 - Math.abs(s.f - tune) / 6);
        ctx.fillStyle = `rgba(255,176,143,${0.35 + 0.65 * near})`;
        ctx.font = `600 ${(12 + near * 3) * dpr}px "Sora Variable", sans-serif`;
        ctx.textAlign = "center";
        ctx.fillText(s.name.toUpperCase(), x(s.f), H * 0.56);
        ctx.fillRect(x(s.f) - dpr, H * 0.4, 2 * dpr, H * 0.06);
      }
      // the signal: a calm wave on a station, noise between
      ctx.strokeStyle = `rgba(255,211,161,${0.85 - off * 0.6})`;
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      for (let px = 0; px <= W; px += 4 * dpr) {
        const y = H * 0.8 + Math.sin(px * 0.02 / dpr + t * 2) * H * 0.07 * (1 - off) * Math.sin(px * 0.006 / dpr + t * 0.6) + (reduce ? 0 : (Math.random() - 0.5) * H * 0.12 * off);
        px ? ctx.lineTo(px, y) : ctx.moveTo(px, y);
      }
      ctx.stroke();
      // the needle
      ctx.strokeStyle = "#ffb08f";
      ctx.lineWidth = 2.5 * dpr;
      ctx.shadowColor = "#ffb08f";
      ctx.shadowBlur = 10 * dpr;
      ctx.beginPath();
      ctx.moveTo(x(tune), H * 0.05);
      ctx.lineTo(x(tune), H * 0.95);
      ctx.stroke();
      ctx.shadowBlur = 0;
      dimple?.style.setProperty("transform", `rotate(${angle}rad)`);
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      audio?.ctx.close();
    };
  });
</script>

<div class="receiver">
  <canvas bind:this={canvas} aria-hidden="true"></canvas>
  <div class="row">
    <div class="readout" aria-live="polite">
      <span class="freq">{(86 + tune * 0.2).toFixed(1)} MHz <span class="lock" class:on={locked}></span><span class="lock-label">{onair && shown?.id === tunedId ? "On air" : locked ? "Tuned" : ""}</span></span>
      {#if shown}
        <span class="name">{shown.name}</span>
        <span class="desc">{shown.description} <span class="genre">{genreName(shown.genre)}</span></span>
      {:else}
        <span class="name dim">Between stations</span>
        <span class="desc">Turn the dial to find one.</span>
      {/if}
    </div>
    <div
      class="dial"
      role="slider"
      tabindex="0"
      aria-label="Tuning dial (arrow keys to tune, Enter to tune in)"
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow={Math.round(tune)}
      aria-valuetext={shown ? shown.name : "between stations"}
      onpointerdown={down}
      onpointermove={move}
      onpointerup={up}
      onpointercancel={up}
      onkeydown={key}
    >
      <span class="dimple" bind:this={dimple}></span>
    </div>
  </div>
</div>

<style>
  .receiver { display: grid; gap: 18px; }
  canvas { display: block; width: 100%; aspect-ratio: 9 / 2.4; border-radius: 16px; background: #0b0a16; box-shadow: inset 0 2px 14px rgba(0, 0, 0, 0.9), 0 0 0 1px #2d2650; }
  .row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 24px; align-items: center; }
  .readout { display: grid; gap: 4px; min-width: 0; }
  .freq { display: flex; align-items: center; gap: 8px; font: 500 12px var(--mono); letter-spacing: 0.1em; color: var(--apricot); }
  .lock { width: 8px; height: 8px; border-radius: 50%; background: #3a2a3a; }
  .lock.on { background: var(--peach); box-shadow: 0 0 10px var(--peach); }
  .lock-label { text-transform: uppercase; color: var(--muted); }
  .name { font: 700 1.5rem/1.2 var(--display); }
  .name.dim { color: var(--muted); }
  .desc { color: var(--muted); font-size: 0.92rem; }
  .genre { font: 500 11px var(--mono); color: var(--violet); margin-left: 0.4rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .dial { width: 112px; height: 112px; border-radius: 50%; position: relative; cursor: grab; touch-action: none; flex: none;
    background: radial-gradient(circle at 40% 32%, #3d355f, #221d3c 65%); box-shadow: 0 8px 22px rgba(0, 0, 0, 0.55), 0 0 0 1px #2d2650, 0 0 0 7px #17132a; }
  .dial:active { cursor: grabbing; }
  .dial:focus-visible { outline: 2px solid var(--accent); outline-offset: 10px; }
  .dimple { position: absolute; width: 13px; height: 13px; border-radius: 50%; left: 50%; top: 12px; margin-left: -6.5px; background: #120f22;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.8), 0 0 8px rgba(255, 176, 143, 0.5); transform-origin: 6.5px 44px; }
  @media (max-width: 480px) { .row { grid-template-columns: minmax(0, 1fr); justify-items: center; text-align: center; } .freq { justify-content: center; } }
</style>
