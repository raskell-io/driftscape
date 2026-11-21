<script>
  // Producer mode: a rack of layers. Each layer is a module (genre, mood, gain, pan, mute) beside a lane of clips on a
  // shared timeline. Clips are takes the voice made: drag to move, drag the right edge to trim, then regenerate, extend
  // (a continuation that picks up where the take ends), duplicate or delete. The preview and the exported WAV are the
  // Rust core's mixdown of the same arrangement (lib/producer/mixdown.js).
  import { onMount } from "svelte";
  import { fill } from "#lib/engine.js";
  import { CancelledError, explain } from "#lib/voice/voice.js";
  import * as store from "#lib/store.js";
  import * as P from "#lib/producer/project.js";
  import { Mixdown, loadCore, stereoWav } from "#lib/producer/mixdown.js";
  import Knob from "./Knob.svelte";

  /** @type {{ genres: any[], voice: () => any, ready: boolean, onclaim: () => void }} */
  let { genres, voice, ready, onclaim } = $props();

  const PREVIEW_RATE = 32000;
  const EXPORT_RATE = 48000;
  const PEAKS_PER_SECOND = 20;
  const TARGET_RMS = 0.1;

  let project = $state(null);
  let peaks = $state({}); // take id -> Float32Array
  let selected = $state(null); // clip id
  let position = $state(0); // playhead, seconds
  let playing = $state(false);
  let busy = $state(null); // { trackId, fraction, what }
  let error = $state("");
  let pps = $state(14); // pixels per second
  let level = $state(8);

  let mix = null;
  let buffer = $state.raw(null);
  let ctx = null;
  let out = null;
  let source = null;
  let started = { at: 0, from: 0 };
  let abort = null;

  const length = $derived(project ? P.projectSeconds(project) : 0);
  const laneSeconds = $derived(Math.max(60, Math.ceil((length + 20) / 10) * 10));
  const genreOf = (id) => genres.find((g) => g.id === id) ?? genres[0];
  const selection = $derived.by(() => {
    if (!project || !selected) return null;
    for (const track of project.tracks) {
      const clip = track.clips.find((c) => c.id === selected);
      if (clip) return { track, clip };
    }
    return null;
  });

  onMount(() => {
    (async () => {
      try {
        mix = new Mixdown(await loadCore());
        const saved = await store.loadProject();
        const p = saved ?? P.newProject();
        for (const id of Object.keys(p.takes)) {
          const take = await store.loadTake(id).catch(() => undefined);
          if (!take) {
            // Lost (storage cleared): drop the clips that played it.
            for (const t of p.tracks) t.clips = t.clips.filter((c) => c.take !== id);
            delete p.takes[id];
            continue;
          }
          mix.load(id, take.channels, take.sampleRate);
          peaks[id] = take.peaks;
        }
        project = p;
        refresh();
      } catch (e) {
        error = `Couldn't open the project: ${e.message}`;
      }
    })();
    return () => {
      abort?.abort();
      halt();
      ctx?.close();
    };
  });

  // ---- after every edit: save, mix down again, keep playing from the same place --------------------------------------

  let timer = null;
  function commit() {
    store.saveProject($state.snapshot(project));
    clearTimeout(timer);
    timer = setTimeout(refresh, 120);
  }

  function refresh() {
    if (!project || !mix) return;
    try {
      const [l, r] = mix.render($state.snapshot(project), PREVIEW_RATE);
      buffer = l.length ? { l, r } : null;
    } catch (e) {
      error = e.message;
      return;
    }
    if (playing) startSource(now());
  }

  // ---- transport -------------------------------------------------------------------------------------------------

  function now() {
    return playing && ctx ? started.from + (ctx.currentTime - started.at) : position;
  }

  function startSource(from) {
    source?.stop();
    source = null;
    if (!buffer || from >= buffer.l.length / PREVIEW_RATE) {
      halt();
      position = buffer && from >= buffer.l.length / PREVIEW_RATE ? 0 : from;
      return;
    }
    const b = new AudioBuffer({ length: buffer.l.length, sampleRate: PREVIEW_RATE, numberOfChannels: 2 });
    b.copyToChannel(buffer.l, 0);
    b.copyToChannel(buffer.r, 1);
    const s = ctx.createBufferSource();
    s.buffer = b;
    s.connect(out);
    s.onended = () => {
      if (source !== s) return; // replaced or stopped on purpose
      halt();
      position = 0;
    };
    s.start(0, from);
    source = s;
    started = { at: ctx.currentTime, from };
  }

  async function play() {
    if (!buffer) return;
    onclaim();
    ctx ??= new AudioContext({ latencyHint: "playback" });
    if (!out) {
      out = ctx.createGain();
      out.connect(ctx.destination);
    }
    out.gain.value = (level / 10) ** 2;
    await ctx.resume();
    playing = true;
    startSource(position);
    const tick = () => {
      if (!playing) return;
      position = now();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function halt() {
    if (playing) position = now();
    playing = false;
    const s = source;
    source = null;
    s?.stop();
  }

  /** Stop and keep the playhead (the parent calls this when live or radio starts). */
  export function stop() {
    halt();
  }

  function seek(e) {
    if (e.target !== e.currentTarget) return;
    const r = e.currentTarget.getBoundingClientRect();
    position = Math.max(0, (e.clientX - r.left) / pps);
    if (playing) startSource(position);
  }

  // ---- generating takes ------------------------------------------------------------------------------------------

  async function makeTake(track, what, continueFrom = null) {
    const v = voice();
    if (!v || busy) return null;
    error = "";
    onclaim();
    const g = genreOf(track.genreId);
    const request = {
      prompt: fill(g, track.prompt),
      seconds: g.segment_seconds,
      guidance: g.guidance,
      temperature: g.temperature,
      topK: 250,
      seed: Math.floor(Math.random() * 2 ** 52),
      continueFrom,
      analysis: { crossfadeSeconds: g.crossfade_seconds, bpm: g.bpm },
    };
    busy = { trackId: track.id, fraction: 0, what };
    abort = new AbortController();
    try {
      const result = await v.generate(request, (f) => busy && (busy.fraction = f), abort.signal);
      const channels = [leveled(result.audio)];
      const id = `take-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
      const take = { id, channels, sampleRate: result.sampleRate, codebooks: result.codebooks ?? null, continuedFrom: continueFrom, peaks: peaksOf(channels[0], result.sampleRate) };
      await store.saveTake(take);
      mix.load(id, channels, take.sampleRate);
      peaks[id] = take.peaks;
      return { id, info: { seconds: channels[0].length / take.sampleRate, prompt: request.prompt, genreId: g.id, overlapSeconds: result.overlapSeconds ?? 0 } };
    } catch (e) {
      if (!(e instanceof CancelledError)) error = explain(e.message);
      return null;
    } finally {
      busy = null;
      abort = null;
    }
  }

  /** A steady level per take, like live mode: about the same loudness, never clipping. */
  function leveled(audio) {
    let sum = 0;
    let peak = 1e-6;
    for (const s of audio) {
      sum += s * s;
      peak = Math.max(peak, Math.abs(s));
    }
    const gain = Math.min(TARGET_RMS / Math.max(Math.sqrt(sum / audio.length), 1e-6), 8, 0.95 / peak);
    return audio.map((s) => s * gain);
  }

  function peaksOf(samples, rate) {
    const n = Math.max(1, Math.ceil((samples.length / rate) * PEAKS_PER_SECOND));
    const per = samples.length / n;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (let j = Math.floor(i * per); j < Math.floor((i + 1) * per); j++) m = Math.max(m, Math.abs(samples[j]));
      out[i] = m;
    }
    return out;
  }

  async function newTake(track) {
    const made = await makeTake(track, "New take");
    if (!made) return;
    const clip = P.appendTake(project, track, made.id, made.info, P.snap(position, genreOf(track.genreId).bpm));
    selected = clip.id;
    commit();
  }

  async function regenerate({ track, clip }) {
    const old = await store.loadTake(clip.take);
    const made = await makeTake(track, "Regenerating", old?.continuedFrom ?? null);
    if (!made) return;
    const unused = P.replaceTake(project, clip, made.id, made.info);
    if (unused) store.deleteTake(unused);
    commit();
  }

  async function extend({ track, clip }) {
    const take = await store.loadTake(clip.take);
    if (!take?.codebooks) {
      error = "This take can't be extended (its voice doesn't continue takes).";
      return;
    }
    const g = genreOf(track.genreId);
    const frames = Math.round(g.overlap_seconds * 50);
    const made = await makeTake(track, "Extending", take.codebooks.map((c) => c.slice(-frames)));
    if (!made) return;
    selected = P.extendClip(project, track, clip, made.id, made.info).id;
    commit();
  }

  function cancel() {
    abort?.abort();
  }

  // ---- editing -----------------------------------------------------------------------------------------------------

  function addLayer() {
    const used = new Set(project.tracks.map((t) => t.genreId));
    const g = genres.find((x) => !used.has(x.id)) ?? genres[0];
    project.tracks.push(P.newTrack({ genreId: g.id }));
    commit();
  }

  function removeLayer(track) {
    for (const id of P.removeTrack(project, track)) store.deleteTake(id);
    if (selection?.track === track) selected = null;
    commit();
  }

  function duplicate({ track, clip }) {
    selected = P.duplicateClip(project, track, clip).id;
    commit();
  }

  function remove({ track, clip }) {
    const unused = P.removeClip(project, track, clip);
    if (unused) store.deleteTake(unused);
    selected = null;
    commit();
  }

  function fade({ clip }, which, delta) {
    const d = P.clipDuration(clip, project.takes[clip.take].seconds);
    clip[which] = Math.round(Math.min(d / 2, Math.max(0, clip[which] + delta)) * 4) / 4;
    commit();
  }

  let drag = null;
  function grab(e, track, clip, kind) {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    selected = clip.id;
    drag = { clip, bpm: genreOf(track.genreId).bpm, kind, x: e.clientX, at: clip.at, length: P.clipDuration(clip, project.takes[clip.take].seconds), moved: false };
  }
  function dragging(e) {
    if (!drag) return;
    const dx = (e.clientX - drag.x) / pps;
    if (Math.abs(e.clientX - drag.x) > 2) drag.moved = true;
    if (drag.kind === "move") drag.clip.at = P.snap(drag.at + dx, drag.bpm);
    else P.trimClip(project, drag.clip, Math.max(0.5, P.snap(drag.length + dx, drag.bpm)));
  }
  function drop() {
    if (drag?.moved) commit();
    drag = null;
  }

  function keys(e) {
    if (!selection || e.target.closest("input, select, [role=slider]")) return;
    const { clip, track } = selection;
    const grid = genreOf(track.genreId).bpm ? 60 / genreOf(track.genreId).bpm : 0.5;
    if (e.key === "Delete" || e.key === "Backspace") remove(selection);
    else if (e.key === "ArrowLeft") clip.at = Math.max(0, clip.at - grid * (e.shiftKey ? 8 : 1));
    else if (e.key === "ArrowRight") clip.at += grid * (e.shiftKey ? 8 : 1);
    else if (e.key === "Escape") selected = null;
    else return;
    e.preventDefault();
    if (e.key.startsWith("Arrow")) commit();
  }

  // ---- export ------------------------------------------------------------------------------------------------------

  function exportWav() {
    try {
      const blob = stereoWav(mix.render($state.snapshot(project), EXPORT_RATE), EXPORT_RATE);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${(project.name || "driftscape").toLowerCase().replace(/[^a-z0-9]+/g, "-")}.wav`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    } catch (e) {
      error = e.message;
    }
  }

  function clearProject() {
    halt();
    for (const id of Object.keys(project.takes)) store.deleteTake(id);
    mix.clear();
    project = P.newProject();
    peaks = {};
    selected = null;
    position = 0;
    commit();
  }

  // ---- drawing -------------------------------------------------------------------------------------------------------

  /** Set CSS variables from script: the Content Security Policy blocks inline style attributes. */
  function vars(node, values) {
    const apply = (v) => Object.entries(v).forEach(([k, x]) => node.style.setProperty(`--${k}`, String(x)));
    apply(values);
    return { update: apply };
  }

  function wave(takeId) {
    const p = peaks[takeId];
    if (!p) return "";
    let top = "M0,0";
    let bottom = "";
    for (let i = 0; i < p.length; i++) {
      const y = Math.min(1, p[i] * 2.2).toFixed(3);
      top += ` L${i + 0.5},${-y}`;
      bottom = ` L${i + 0.5},${y}` + bottom;
    }
    return `${top} L${p.length},0${bottom} Z`;
  }

  const clock = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
  const ticks = $derived(Array.from({ length: Math.floor(laneSeconds / 5) + 1 }, (_, i) => i * 5));
</script>

<svelte:window onkeydown={keys} />

{#if project}
  <div class="rack">
    <div class="transport">
      {#if playing}
        <button class="play stop" onclick={halt} aria-label="Stop">■</button>
      {:else}
        <button class="play" onclick={play} disabled={!buffer} aria-label="Play">▶</button>
      {/if}
      <div class="counter" aria-label="Playhead">
        <span class="time">{clock(position)}</span><span class="of">/ {clock(length)}</span>
      </div>
      <label class="name">
        <span class="label">Piece</span>
        <input bind:value={project.name} onchange={commit} autocomplete="off" />
      </label>
      <div class="zoom" aria-label="Zoom">
        <button class="mini" onclick={() => (pps = Math.max(4, pps / 1.5))} aria-label="Zoom out">−</button>
        <button class="mini" onclick={() => (pps = Math.min(80, pps * 1.5))} aria-label="Zoom in">+</button>
      </div>
      <Knob label="Level" size={40} min={0} max={10} bind:value={level} onchange={(v) => out && (out.gain.value = (v / 10) ** 2)} />
      <button class="pill" onclick={exportWav} disabled={!buffer}>Export WAV</button>
    </div>

    <div class="board" use:vars={{ lane: `${laneSeconds * pps}px`, head: `${position * pps}px` }}>
      <div class="row ruler-row">
        <div class="module ruler-corner"><span class="label">Layers</span></div>
        <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
        <div class="lane ruler" onclick={seek}>
          {#each ticks as t (t)}
            <span class="tick" use:vars={{ x: `${t * pps}px` }}>{t % 60 === 0 ? `${t / 60}:00` : `${t}s`}</span>
          {/each}
          <span class="head" aria-hidden="true"></span>
        </div>
      </div>

      {#each project.tracks as track (track.id)}
        {@const g = genreOf(track.genreId)}
        <div class="row" class:muted={track.mute}>
          <div class="module">
            <div class="module-top">
              <select bind:value={track.genreId} onchange={commit} aria-label="Genre">
                {#each genres as x (x.id)}<option value={x.id}>{x.name}</option>{/each}
              </select>
              <button class="mini" class:on={track.mute} aria-pressed={track.mute} onclick={() => ((track.mute = !track.mute), commit())}>M</button>
              <button class="mini" onclick={() => removeLayer(track)} aria-label="Remove layer" disabled={busy?.trackId === track.id}>✕</button>
            </div>
            <input class="mood" bind:value={track.prompt} onchange={commit} placeholder={g.default_prompt} autocomplete="off" aria-label="Mood" />
            <div class="module-bottom">
              <Knob label="Gain" size={34} min={-24} max={6} bind:value={track.gainDb} format={(v) => `${v} dB`} onchange={commit} />
              <Knob label="Pan" size={34} min={-10} max={10} value={Math.round(track.pan * 10)} format={(v) => (v ? (v < 0 ? `L${-v}` : `R${v}`) : "C")} onchange={(v) => ((track.pan = v / 10), commit())} />
              {#if busy?.trackId === track.id}
                <button class="pill take busy" onclick={cancel} title="Cancel">{busy.what} · {Math.round(busy.fraction * 100)} %</button>
              {:else}
                <button class="pill take" onclick={() => newTake(track)} disabled={!ready || !!busy}>New take</button>
              {/if}
            </div>
          </div>
          <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
          <div class="lane" onclick={seek}>
            {#each track.clips as clip (clip.id)}
              {@const info = project.takes[clip.take]}
              {@const d = P.clipDuration(clip, info?.seconds ?? 0)}
              <div
                class="clip"
                class:selected={selected === clip.id}
                role="button"
                tabindex="0"
                aria-label={`Clip at ${clock(clip.at)}, ${d.toFixed(1)} s`}
                use:vars={{ x: `${clip.at * pps}px`, w: `${d * pps}px`, fi: `${clip.fadeIn * pps}px`, fo: `${clip.fadeOut * pps}px` }}
                onpointerdown={(e) => grab(e, track, clip, "move")}
                onpointermove={dragging}
                onpointerup={drop}
                onpointercancel={drop}
              >
                <svg viewBox={`${clip.offset * PEAKS_PER_SECOND} -1 ${Math.max(0.01, d * PEAKS_PER_SECOND)} 2`} preserveAspectRatio="none" aria-hidden="true">
                  <path d={wave(clip.take)} />
                </svg>
                <span class="fade in" aria-hidden="true"></span>
                <span class="fade out" aria-hidden="true"></span>
                <span
                  class="trim"
                  role="separator"
                  aria-label="Trim"
                  onpointerdown={(e) => grab(e, track, clip, "trim")}
                  onpointermove={dragging}
                  onpointerup={drop}
                  onpointercancel={drop}
                ></span>
              </div>
            {/each}
            <span class="head" aria-hidden="true"></span>
          </div>
        </div>
      {/each}

      <div class="row">
        <div class="module add">
          <button class="pill" onclick={addLayer} disabled={!genres.length}>+ Layer</button>
        </div>
        <div class="lane empty">
          {#if !project.tracks.length}<p class="hint">Add a layer, pick its genre and mood, and make a take.</p>{/if}
        </div>
      </div>
    </div>

    {#if selection}
      {@const s = selection}
      <div class="inspector" aria-label="Selected clip">
        <span class="label">Clip</span>
        <span class="what">{project.takes[s.clip.take]?.prompt}</span>
        <div class="actions">
          <button class="pill" onclick={() => regenerate(s)} disabled={!ready || !!busy}>Regenerate</button>
          <button class="pill" onclick={() => extend(s)} disabled={!ready || !!busy}>Extend</button>
          <button class="pill" onclick={() => duplicate(s)}>Duplicate</button>
          <span class="fades">
            Fade in <button class="mini" onclick={() => fade(s, "fadeIn", -0.25)}>−</button>{s.clip.fadeIn.toFixed(2)} s<button class="mini" onclick={() => fade(s, "fadeIn", 0.25)}>+</button>
          </span>
          <span class="fades">
            out <button class="mini" onclick={() => fade(s, "fadeOut", -0.25)}>−</button>{s.clip.fadeOut.toFixed(2)} s<button class="mini" onclick={() => fade(s, "fadeOut", 0.25)}>+</button>
          </span>
          <button class="link" onclick={() => remove(s)}>Delete</button>
        </div>
      </div>
    {:else if project.tracks.some((t) => t.clips.length)}
      <p class="hint">Drag clips to move them, their right edge to trim. Select one to regenerate or extend it. Takes stay on this device.
        <button class="link" onclick={clearProject} disabled={!!busy}>Start a new piece</button></p>
    {/if}
    {#if error}<p class="error" role="alert">{error}</p>{/if}
  </div>
{:else if error}
  <p class="error" role="alert">{error}</p>
{/if}

<style>
  .rack { display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; min-width: 0; }
  .transport { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .play { width: 48px; height: 48px; flex: none; border-radius: 50%; border: 0; background: var(--peach); color: #24204a; font-size: 1.05rem;
    cursor: pointer; box-shadow: 0 6px 20px rgba(255, 176, 143, 0.35); }
  .play:disabled { opacity: 0.35; cursor: default; box-shadow: none; }
  .play.stop { background: var(--raised); color: var(--ink); box-shadow: none; border: 1px solid var(--line); }
  .counter { padding: 0.5rem 0.8rem; border-radius: 10px; background: #0b0a16; box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.8);
    font: 500 15px var(--mono); color: var(--apricot); display: flex; gap: 0.5rem; align-items: baseline; }
  .counter .of { font-size: 11px; color: var(--muted); }
  .name { display: flex; align-items: center; gap: 8px; flex: 1; min-width: 160px; padding: 0 12px; border-radius: 12px; background: #120f22;
    box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.7); }
  .name .label { margin: 0; }
  .name input { flex: 1; min-width: 0; border: 0; outline: 0; background: none; padding: 0.6rem 0; font: 600 0.95rem var(--display); }
  .zoom { display: flex; gap: 4px; }

  .board { --module: 236px; overflow-x: auto; border-radius: 16px; background: #0b0a16; box-shadow: inset 0 2px 14px rgba(0, 0, 0, 0.9), 0 0 0 1px #2d2650; }
  .row { display: flex; min-width: calc(var(--module) + var(--lane)); border-bottom: 1px solid #1d1933; }
  .module { position: sticky; left: 0; z-index: 2; flex: none; width: var(--module); box-sizing: border-box; padding: 10px; display: grid; gap: 8px;
    background: linear-gradient(180deg, #1d1934, #17142a); border-right: 1px solid #2d2650; }
  .ruler-corner { padding-block: 6px; align-content: center; }
  .module-top { display: flex; gap: 6px; align-items: center; }
  .module-top select { flex: 1; min-width: 0; background: #120f22; color: var(--ink); border: 1px solid var(--line); border-radius: 8px;
    padding: 0.3rem 0.4rem; font: 600 12px var(--display); }
  .mood { background: #120f22; border: 1px solid var(--line); border-radius: 8px; padding: 0.35rem 0.5rem; font-size: 0.85rem; outline: 0; }
  .mood:focus { border-color: var(--accent); }
  .module-bottom { display: flex; align-items: center; gap: 10px; }
  .module-bottom :global(.knob) { gap: 3px; }
  .module-bottom :global(.name) { font-size: 8px; }
  .module-bottom :global(.value) { font-size: 10px; }
  .take { margin-left: auto; font-size: 0.8rem; padding: 0.4rem 0.7rem; }
  .take.busy { border-color: var(--peach); color: var(--apricot); font: 500 11px var(--mono); }
  .add { align-content: center; }
  .row.muted .lane { opacity: 0.35; }

  .lane { position: relative; flex: none; width: var(--lane); min-height: 132px; cursor: text;
    background-image: linear-gradient(90deg, rgba(169, 147, 240, 0.06) 1px, transparent 1px); background-size: calc(var(--lane) / 12) 100%; }
  .lane.empty { min-height: 64px; display: flex; align-items: center; padding-left: 16px; cursor: default; }
  .ruler { min-height: 26px; cursor: pointer; background: none; }
  .tick { position: absolute; left: var(--x); top: 6px; padding-left: 4px; border-left: 1px solid #3a3360; font: 500 10px var(--mono); color: var(--muted);
    pointer-events: none; }
  .head { position: absolute; top: 0; bottom: 0; left: var(--head); width: 2px; margin-left: -1px; background: var(--peach); pointer-events: none;
    box-shadow: 0 0 8px rgba(255, 176, 143, 0.6); z-index: 1; }

  .clip { position: absolute; top: 12px; bottom: 12px; left: var(--x); width: var(--w); box-sizing: border-box; border-radius: 8px; overflow: hidden;
    cursor: grab; touch-action: none; background: rgba(124, 99, 210, 0.22); border: 1px solid rgba(169, 147, 240, 0.45); }
  .clip:active { cursor: grabbing; }
  .clip.selected { border-color: var(--peach); background: rgba(255, 176, 143, 0.14); box-shadow: 0 0 0 1px var(--peach), 0 0 18px rgba(255, 176, 143, 0.25); }
  .clip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .clip svg { position: absolute; inset: 6px 0; width: 100%; height: calc(100% - 12px); pointer-events: none; }
  .clip path { fill: rgba(255, 211, 161, 0.55); }
  .clip.selected path { fill: rgba(255, 211, 161, 0.85); }
  .fade { position: absolute; top: 0; bottom: 0; pointer-events: none; }
  .fade.in { left: 0; width: var(--fi); background: linear-gradient(to top right, rgba(11, 10, 22, 0.55) 50%, transparent 50%); }
  .fade.out { right: 0; width: var(--fo); background: linear-gradient(to top left, rgba(11, 10, 22, 0.55) 50%, transparent 50%); }
  .trim { position: absolute; right: 0; top: 0; bottom: 0; width: 10px; cursor: ew-resize; background: linear-gradient(90deg, transparent, rgba(255, 211, 161, 0.35)); }

  .inspector { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 10px 12px; border-radius: 12px; background: #120f22;
    border: 1px solid var(--line); }
  .inspector .label { margin: 0; }
  .what { flex: 1; min-width: 160px; font-size: 0.85rem; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .fades { display: inline-flex; align-items: center; gap: 4px; font: 500 11px var(--mono); color: var(--muted); }

  .mini { min-width: 26px; height: 26px; border-radius: 7px; border: 1px solid var(--line); background: var(--raised); color: var(--ink); cursor: pointer;
    font: 600 12px var(--mono); }
  .mini.on { background: var(--peach); color: #24204a; border-color: var(--peach); }
  .mini:disabled { opacity: 0.4; cursor: default; }
  .pill { border: 1px solid var(--line); background: var(--raised); border-radius: 999px; padding: 0.45rem 0.9rem; font-weight: 600; cursor: pointer; }
  .pill:disabled { opacity: 0.4; cursor: default; }
  .label { font: 500 11px/1.4 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  .link { background: none; border: 0; padding: 0; color: var(--muted); text-decoration: underline; cursor: pointer; font-size: 0.85rem; }
  .hint { margin: 0; color: var(--muted); font-size: 0.88rem; }
  .error { color: var(--rose); margin: 0; }

  @media (max-width: 680px) {
    .board { --module: 168px; }
    .module-bottom { flex-wrap: wrap; }
  }
</style>
