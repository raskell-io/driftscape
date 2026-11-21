<script>
  import { onMount } from "svelte";
  import { chooseVoice } from "#lib/voice/voice.js";
  import { LiveEngine } from "#lib/engine.js";
  import * as store from "#lib/store.js";
  import { blendAt, differs, regions } from "#lib/ui/terrain.js";
  import Terrain from "#lib/ui/Terrain.svelte";
  import Knob from "#lib/ui/Knob.svelte";
  import Receiver from "#lib/ui/Receiver.svelte";

  let presets = $state({ genres: [], stations: [] });
  let mode = $state("live"); // "live" | "radio"
  let dot = $state({ x: 0.36, y: 0.5 });
  let mix = $state(null); // { genre, blend } at the dot
  let prompt = $state("");
  let landmarks = $state([]);
  let focus = $state(0); // -2..2, steps of 0.5 guidance
  let wander = $state(0); // -2..2, steps of 0.1 temperature
  let drift = $state(3); // 0..10, the terrain's autopilot
  let level = $state(8); // 0..10
  let stationId = $state(null);
  let settingsLoaded = $state(false);

  let engineState = $state({ phase: "starting", detail: "Looking for an engine", fraction: 0 });
  let voiceLabel = $state("");
  let native = $state(false);
  let playing = $state(false);
  let nowPlaying = $state("");
  let progress = $state(0);
  let startedAt = $state(0);
  let now = $state(Date.now());
  let slow = $state(false);
  let log = $state([]);
  let error = $state("");
  let recording = $state(false);
  let recordedSeconds = $state(0);

  let resumable = $state(null);
  let recordings = $state([]);
  let storageInfo = $state(null);
  let updateReady = $state(null);
  let online = $state(true);

  let engine = null;
  let recordTimer = null;
  let sent = null; // the mix the engine is playing

  const engineReady = $derived(engineState.phase === "ready");
  const live = $derived(playing && !stationId);
  const station = $derived(presets.stations.find((s) => s.id === stationId));
  const rs = $derived(regions(presets.genres));
  const waitingSeconds = $derived(startedAt && !nowPlaying ? Math.max(0, (now - startedAt) / 1000) : 0);
  const pulseBpm = $derived(
    live ? (mix?.genre.bpm ?? null) : playing ? (presets.genres.find((g) => g.id === station?.genre)?.bpm ?? null) : null,
  );

  onMount(async () => {
    online = navigator.onLine;
    addEventListener("online", () => (online = true));
    addEventListener("offline", () => (online = false));
    setInterval(() => (now = Date.now()), 1000);
    watchForUpdates();

    const [loaded, settings, session] = await Promise.all([
      fetch("/presets.json").then((r) => r.json()),
      store.loadSettings(),
      store.loadSession(),
    ]);
    presets = loaded;
    if (settings) {
      ({ mode = "live", prompt = "", focus = 0, wander = 0, drift = 3, level = 8 } = settings);
      if (settings.dot) dot = settings.dot;
      if (settings.landmarks) landmarks = settings.landmarks;
    }
    mix = blendAt(regions(presets.genres), dot.x, dot.y);
    settingsLoaded = true;
    if (session?.audio) resumable = session;
    refreshLibrary();

    const voice = await chooseVoice(({ detail, fraction }) => (engineState = { phase: "loading", detail, fraction }));
    voiceLabel = voice.label;
    native = voice.label.includes("engine");
    engine = new LiveEngine(voice, onEvent);
    engine.setLevel(level / 10);
    try {
      await voice.ready();
      engineState = { phase: "ready", detail: "Ready", fraction: 1 };
      if (!native) await store.persist();
    } catch (e) {
      engineState = { phase: "failed", detail: e.message, fraction: 0 };
    }
    refreshLibrary();
  });

  $effect(() => {
    const settings = { mode, prompt, focus, wander, drift, level, dot: { ...dot }, landmarks: landmarks.map((l) => ({ ...l })) };
    if (settingsLoaded) store.saveSettings(settings);
  });

  async function refreshLibrary() {
    recordings = await store.listRecordings().catch(() => []);
    storageInfo = await store.storage().catch(() => null);
  }

  let updateRequested = false;
  async function watchForUpdates() {
    const reg = await navigator.serviceWorker?.getRegistration();
    if (!reg) return;
    const offer = (w) => {
      if (w && navigator.serviceWorker.controller) updateReady = w;
    };
    offer(reg.waiting);
    reg.addEventListener("updatefound", () => {
      const w = reg.installing;
      w?.addEventListener("statechange", () => w.state === "installed" && offer(w));
    });
    // Reload only for an update the listener asked for (a first install also changes the controller).
    navigator.serviceWorker.addEventListener("controllerchange", () => updateRequested && location.reload());
    setInterval(() => reg.update().catch(() => {}), 30 * 60 * 1000);
  }
  function applyUpdate() {
    updateRequested = true;
    updateReady?.postMessage({ type: "skip-waiting" });
  }

  function addLog(entry) {
    log = [entry, ...log].slice(0, 5);
  }

  function onEvent(e) {
    if (e.type === "segment") {
      nowPlaying = e.label;
      if (e.speed != null) slow = !native && e.speed < 0.3;
      addLog({ kind: "segment", text: e.label });
    } else if (e.type === "repeat") {
      const count = log[0]?.kind === "repeat" ? log[0].count + 1 : 1;
      const entry = { kind: "repeat", count, text: `Repeating while the next part is made${count > 1 ? ` ×${count}` : ""}` };
      log = log[0]?.kind === "repeat" ? [entry, ...log.slice(1)] : [entry, ...log].slice(0, 5);
    } else if (e.type === "progress") {
      progress = e.fraction;
    } else if (e.type === "state") {
      playing = e.playing;
      if (!e.playing) nowPlaying = "";
    } else if (e.type === "snapshot") {
      store.saveSession(e.snapshot);
    } else if (e.type === "error") {
      error = e.message;
    }
  }

  function begin() {
    error = "";
    log = [];
    nowPlaying = "";
    progress = 0;
    startedAt = Date.now();
    resumable = null;
  }

  // ---- live: the terrain ----

  async function playLive() {
    begin();
    stationId = null;
    sent = mix;
    await engine.start({ genre: mix.genre, blend: mix.blend, prompt, focus: focus * 0.5, variety: wander * 0.1 });
  }

  /** The dot moved. By hand, the change comes in right away; on autopilot it shapes the next part. */
  function onmove({ x, y, mix: m, byHand, settled }) {
    dot = { x, y };
    mix = m;
    if (!live || !settled || !differs(sent, m)) return;
    const arrived = sent?.genre.id !== m.genre.id;
    sent = m;
    engine.tweak({ genre: m.genre, blend: m.blend }, { urgent: byHand });
    // Say so when the dot reaches a new region; small shifts within one are heard, not announced.
    if (arrived) addLog({ kind: "drift", text: `${byHand ? "Moving" : "Drifting"} into ${m.genre.name}` });
  }

  /** Fly the dot to a region (the legend). */
  function visit(r) {
    dot = { x: r.x, y: r.y + 0.04 };
    onmove({ ...dot, mix: blendAt(rs, dot.x, dot.y), byHand: true, settled: true });
  }

  function submitMood(event) {
    event.preventDefault();
    const words = prompt.trim();
    if (words) {
      landmarks = [...landmarks.slice(-3), { x: dot.x, y: dot.y, word: words.split(/\s+/).slice(0, 3).join(" ") }];
    }
    if (live) engine.tweak({ prompt });
    else if (engineReady) playLive();
  }

  const steer = (change) => live && engine.tweak(change);

  // ---- radio: the receiver ----

  async function tune(id) {
    if (stationId === id && playing) return;
    begin();
    stationId = id;
    const s = presets.stations.find((x) => x.id === id);
    await engine.start({ genre: presets.genres.find((g) => g.id === s.genre), station: s });
  }

  function stop() {
    engine?.stop();
    stationId = null;
    startedAt = 0;
  }

  // ---- resume, recording ----

  async function resume() {
    const snap = resumable;
    const s = presets.stations.find((x) => x.id === snap.stationId) ?? null;
    const g = presets.genres.find((x) => x.id === snap.params.genreId) ?? mix.genre;
    const b = presets.genres.find((x) => x.id === snap.params.blendId);
    begin();
    stationId = s?.id ?? null;
    mode = s ? "radio" : "live";
    if (!s) prompt = snap.params.prompt;
    sent = s ? null : { genre: g, blend: b ? { genre: b, share: snap.params.blendShare } : null };
    await engine.start({
      genre: g,
      blend: sent?.blend ?? null,
      prompt: snap.params.prompt,
      station: s,
      focus: snap.params.focus,
      variety: snap.params.variety,
      resume: snap,
    });
  }
  function dismissResume() {
    resumable = null;
    store.clearSession();
  }

  async function toggleRecording() {
    if (!recording) {
      recording = true;
      const label =
        station?.name ??
        (mix ? `${mix.genre.name}${mix.blend ? ` × ${mix.blend.genre.name}` : ""}${prompt.trim() ? ` · ${prompt.trim()}` : ""}` : "Live");
      await engine.startRecording(label);
      recordTimer = setInterval(() => (recordedSeconds = engine.recordedSeconds()), 500);
      refreshLibrary();
      return;
    }
    clearInterval(recordTimer);
    recording = false;
    recordedSeconds = 0;
    await engine.stopRecording();
    refreshLibrary();
  }
  async function saveRecording(rec) {
    const blob = await store.recordingWav(rec.id);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `driftscape-${new Date(rec.startedAt).toISOString().slice(0, 16).replace(/[-:T]/g, "")}.wav`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
  }
  async function deleteRecording(rec) {
    await store.deleteRecording(rec.id);
    refreshLibrary();
  }
  async function removeModel() {
    await store.removeModel();
    refreshLibrary();
  }

  const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const mb = (bytes) => `${Math.round(bytes / 1e6)} MB`;
  const signed = (v) => (v > 0 ? `+${v}` : String(v));
  function ago(ms) {
    const m = Math.round((Date.now() - ms) / 60000);
    if (m < 1) return "just now";
    if (m < 60) return `${m} min ago`;
    const h = Math.round(m / 60);
    return h < 24 ? `${h} h ago` : new Date(ms).toLocaleDateString();
  }
  const when = (ms) => new Date(ms).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
</script>

<div class="band" aria-hidden="true"></div>

<header class="top">
  <div class="brand">
    <img class="icon" src="/icon.svg" alt="" width="36" height="36" />
    <h1><img class="wordmark" src="/wordmark-dark.svg" alt="Driftscape" /></h1>
  </div>
  <p class="engine" class:ready={engineReady} class:failed={engineState.phase === "failed"} role="status">
    {#if engineState.phase === "failed"}{engineState.detail}
    {:else}<span class="lamp"></span>{engineReady ? "Engine ready" : engineState.detail}{engineState.phase === "loading" &&
      engineState.fraction > 0 &&
      engineState.fraction < 1
        ? ` · ${Math.round(engineState.fraction * 100)} %`
        : ""}{/if}
  </p>
</header>

{#if engineState.phase === "loading"}
  <div class="loadbar" aria-hidden="true"><span style:--f={engineState.fraction}></span></div>
  <p class="hint">The first visit downloads the model once (about 570 MB). After that it starts in seconds, even offline.</p>
{/if}
{#if updateReady}
  <div class="banner" role="status">
    <span>A new version of Driftscape is ready.</span>
    <button class="pill" onclick={applyUpdate}>{playing ? "Reload (stops the music)" : "Reload"}</button>
  </div>
{/if}
{#if !online}
  <div class="banner quiet" role="status">
    {#if storageInfo?.modelCached || native}You're offline. Everything still works: the music is made on this device.
    {:else}You're offline, and the model isn't downloaded yet. Connect once to fetch it.{/if}
  </div>
{/if}
{#if resumable && !playing}
  <section class="resume" aria-label="Continue where you left off">
    <div>
      <span class="label">Continue where you left off</span>
      <p class="resume-text">{resumable.label}</p>
      <p class="resume-when">{ago(resumable.savedAt)} · plays at once and carries on from there</p>
    </div>
    <div class="resume-actions">
      <button class="play small" onclick={resume} disabled={!engineReady} aria-label="Continue">▶</button>
      <button class="link" onclick={dismissResume}>Dismiss</button>
    </div>
  </section>
{/if}

<section class="instrument" aria-label="Driftscape">
  <div class="bar">
    <div class="modes" role="tablist" aria-label="Mode">
      <button role="tab" aria-selected={mode === "live"} class:active={mode === "live"} onclick={() => (mode = "live")}>Live<small>Terrain</small></button>
      <button role="tab" aria-selected={mode === "radio"} class:active={mode === "radio"} onclick={() => (mode = "radio")}>Radio<small>Receiver</small></button>
    </div>
    <button class="rec" class:on={recording} onclick={toggleRecording} disabled={!playing && !recording} aria-pressed={recording}>
      <span class="rec-dot"></span>{recording ? `Stop recording · ${clock(recordedSeconds)}` : "Record"}
    </button>
  </div>

  {#if mode === "live"}
    <div class="terrain-grid">
      <div class="knobs">
        <Knob label="Focus" min={-2} max={2} bind:value={focus} format={signed} onchange={(v) => steer({ focus: v * 0.5 })} />
        <Knob label="Wander" min={-2} max={2} bind:value={wander} format={signed} onchange={(v) => steer({ variety: v * 0.1 })} />
      </div>
      {#if presets.genres.length}
        <Terrain
          genres={presets.genres}
          x={dot.x}
          y={dot.y}
          {drift}
          {landmarks}
          {playing}
          bpm={pulseBpm}
          loudness={() => engine?.loudness() ?? 0}
          {onmove}
        />
      {/if}
      <div class="knobs">
        <Knob label="Drift" min={0} max={10} bind:value={drift} format={(v) => (v ? String(v) : "off")} />
        <Knob label="Level" min={0} max={10} bind:value={level} onchange={(v) => engine?.setLevel(v / 10)} />
      </div>
    </div>

    <div class="legend" aria-label="Go to a genre">
      {#each rs as r (r.genre.id)}
        <button class="place" class:here={mix?.genre.id === r.genre.id} onclick={() => visit(r)}>{r.genre.name}</button>
      {/each}
    </div>

    <form class="deck" onsubmit={submitMood}>
      {#if live}
        <button type="button" class="play stop" onclick={stop} aria-label="Stop">■</button>
      {:else}
        <button type="button" class="play" onclick={playLive} disabled={!engineReady || !mix} aria-label="Play">▶</button>
      {/if}
      <label class="mood" for="mood">
        <span class="label">Mood</span>
        <input id="mood" type="text" bind:value={prompt} placeholder={mix?.genre.default_prompt ?? ""} autocomplete="off" enterkeyhint="go" />
      </label>
      <button type="submit" class="pill" disabled={!engineReady}>{live ? "Steer" : "Play"}</button>
    </form>
  {:else}
    <Receiver stations={presets.stations} genres={presets.genres} tunedId={stationId} onair={playing && !!stationId} onlock={tune} />
    {#if playing && stationId}
      <div class="deck">
        <button class="play stop" onclick={stop} aria-label="Stop">■</button>
        <p class="deck-hint">Turn the dial to change station.</p>
      </div>
    {:else}
      <p class="deck-hint">Turn the dial and rest on a station to tune in.</p>
    {/if}
  {/if}

  <div class="status" aria-live="polite">
    {#if playing}
      <p class="now">
        {nowPlaying ||
          `${station ? `Tuning in to ${station.name}` : "Making the first part"}… ${Math.round(progress * 100)} % · ${Math.floor(waitingSeconds)} s`}
      </p>
      <div class="genbar" aria-hidden="true"><span style:--f={progress}></span></div>
      <ol class="log">
        {#each log.slice(1) as entry, i (i)}<li class:dim={entry.kind !== "segment"}>{entry.text}</li>{/each}
      </ol>
      {#if slow}
        <p class="hint">
          This browser makes music slower than it plays, so parts repeat while the next is made. For smooth, endless play, run
          <code>driftscape serve</code> on this computer.
        </p>
      {/if}
    {:else if engineReady}
      {#if mode === "live"}<p class="now dim">Drag the dot to a place, then press play. Drift makes it wander.</p>{/if}
    {/if}
  </div>
</section>

{#if error}<p class="error" role="alert">{error}</p>{/if}

{#if recordings.length}
  <section class="library" aria-labelledby="rec-title">
    <h2 id="rec-title" class="label">Your recordings</h2>
    <ul>
      {#each recordings as rec (rec.id)}
        <li>
          <div class="rec-info">
            <span class="rec-label">{rec.label}</span>
            <span class="rec-meta"
              >{when(rec.startedAt)} · {clock(rec.frames / rec.sampleRate)}{rec.done ? "" : recording ? " · recording" : " · interrupted"}</span
            >
          </div>
          <div class="rec-actions">
            <button class="pill" onclick={() => saveRecording(rec)} disabled={!rec.frames}>Save WAV</button>
            <button class="link" onclick={() => deleteRecording(rec)} disabled={recording && !rec.done}>Delete</button>
          </div>
        </li>
      {/each}
    </ul>
  </section>
{/if}

<footer>
  {#if voiceLabel}<p>Music is generated by MusicGen small, running on {voiceLabel}. Nothing leaves your device.</p>{/if}
  <p>MusicGen's weights are licensed CC-BY-NC 4.0: free for personal, non-commercial use.</p>
  {#if storageInfo && !native}
    <p>
      {storageInfo.modelCached ? "The model is saved on this device" : "The model isn't downloaded yet"}{storageInfo.persisted ? " and kept" : ""} ·
      {mb(storageInfo.used)} used{#if storageInfo.modelCached && !playing}
        · <button class="link" onclick={removeModel}>Remove the model</button>{/if}
    </p>
  {/if}
</footer>

<style>
  .band { height: 4px; background: var(--dusk); margin: 0 calc(-1 * max(16px, env(safe-area-inset-left, 0px))); }
  .top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; padding-block: 1.25rem 1rem; }
  .brand { display: flex; align-items: center; gap: 0.7rem; }
  .icon { border-radius: 9px; }
  h1 { margin: 0; line-height: 0; }
  .wordmark { height: 26px; width: auto; }
  .engine { margin: 0; display: flex; align-items: center; gap: 0.5rem; font: 500 11px/1.4 var(--mono); letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); }
  .engine.failed { color: var(--rose); text-transform: none; letter-spacing: 0; font: 14px var(--body); }
  .lamp { width: 8px; height: 8px; border-radius: 50%; background: var(--violet); animation: pulse 1.6s ease-in-out infinite; }
  .engine.ready .lamp { background: var(--peach); box-shadow: 0 0 8px var(--peach); animation: none; }
  @keyframes pulse { 50% { opacity: 0.3; } }
  .hint { color: var(--muted); font-size: 0.88rem; margin: 0.5rem 0; }
  .loadbar, .genbar { height: 3px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .loadbar span, .genbar span { display: block; height: 100%; width: calc(var(--f, 0) * 100%); background: var(--dusk); transition: width 0.3s; }

  .banner { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin: 0.75rem 0;
    padding: 0.65rem 1rem; border-radius: 12px; background: var(--raised); border: 1px solid var(--accent); font-size: 0.92rem; }
  .banner.quiet { border-color: var(--line); color: var(--muted); }
  .resume { display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin: 0.75rem 0; padding: 0.9rem 1.1rem;
    border-radius: 14px; background: var(--surface); border: 1px solid var(--line); }
  .resume-text { margin: 0.2rem 0 0; font: 600 1rem/1.4 var(--display); }
  .resume-when { margin: 0.1rem 0 0; color: var(--muted); font-size: 0.85rem; }
  .resume-actions { display: flex; align-items: center; gap: 0.75rem; flex: none; }

  /* the instrument */
  .instrument { margin-top: 0.75rem; padding: 18px; border-radius: 22px; display: grid; gap: 16px;
    background: linear-gradient(180deg, #221d3b, #18142c);
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.06) inset, 0 30px 60px -30px rgba(0, 0, 0, 0.8), 0 0 0 1px #2d2650; }
  .bar { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
  .modes { display: inline-flex; padding: 4px; border-radius: 999px; background: #120f22; box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.7); }
  .modes button { display: flex; align-items: baseline; gap: 0.45rem; border: 0; background: none; padding: 0.45rem 1rem; border-radius: 999px;
    cursor: pointer; font: 600 0.92rem var(--display); color: var(--muted); }
  .modes button small { font: 500 10px var(--mono); letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.7; }
  .modes button.active { background: var(--raised); color: var(--ink); box-shadow: 0 1px 0 rgba(255, 255, 255, 0.06) inset; }
  .rec { display: inline-flex; align-items: center; gap: 0.5rem; border: 1px solid var(--line); background: none; border-radius: 999px;
    padding: 0.4rem 0.9rem; font: 500 12px var(--mono); color: var(--muted); cursor: pointer; }
  .rec:disabled { opacity: 0.4; cursor: default; }
  .rec-dot { width: 9px; height: 9px; border-radius: 50%; background: #5a2a32; }
  .rec.on { color: #ff8a8f; border-color: #ff5a5f; }
  .rec.on .rec-dot { background: #ff5a5f; box-shadow: 0 0 10px #ff5a5f; animation: pulse 1.2s ease-in-out infinite; }

  .terrain-grid { display: grid; grid-template-columns: 84px minmax(0, 1fr) 84px; gap: 14px; align-items: center; }
  .knobs { display: flex; flex-direction: column; gap: 26px; align-items: center; }
  @media (max-width: 680px) {
    .terrain-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .terrain-grid > :global(.screen) { grid-column: 1 / -1; order: -1; }
    .knobs { flex-direction: row; justify-content: space-around; }
  }
  .legend { display: flex; flex-wrap: wrap; gap: 6px; }
  .place { border: 1px solid var(--line); background: none; border-radius: 999px; padding: 0.3rem 0.75rem; cursor: pointer;
    font: 500 12px var(--display); color: var(--muted); }
  .place.here { color: var(--ink); border-color: var(--peach); }

  .deck { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .play { width: 56px; height: 56px; flex: none; border-radius: 50%; border: 0; background: var(--peach); color: #24204a; font-size: 1.2rem;
    cursor: pointer; box-shadow: 0 6px 20px rgba(255, 176, 143, 0.35); }
  .play:disabled { opacity: 0.35; cursor: default; box-shadow: none; }
  .play.stop { background: var(--raised); color: var(--ink); box-shadow: none; border: 1px solid var(--line); }
  .play.small { width: 42px; height: 42px; font-size: 0.95rem; }
  .mood { flex: 1; min-width: 200px; display: flex; align-items: center; gap: 10px; padding: 0 14px; border-radius: 14px; background: #120f22;
    box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.7); }
  .mood .label { margin: 0; }
  .mood input { flex: 1; min-width: 0; border: 0; outline: 0; background: none; padding: 0.85rem 0; font-size: 1rem; }
  .mood:focus-within { box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.7), 0 0 0 2px var(--accent); }
  .pill { border: 1px solid var(--line); background: var(--raised); border-radius: 999px; padding: 0.5rem 1rem; font-weight: 600; cursor: pointer; }
  .pill:disabled { opacity: 0.4; cursor: default; }
  .deck-hint { margin: 0; color: var(--muted); font-size: 0.92rem; }

  .status { display: grid; gap: 6px; min-height: 2.5rem; }
  .now { margin: 0; font: 500 13px/1.5 var(--mono); color: var(--apricot); }
  .now.dim { color: var(--muted); font-family: var(--body); font-size: 0.92rem; }
  .log { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; font: 12px/1.5 var(--mono); color: var(--muted); }
  .log li.dim { color: var(--violet); }

  .label { font: 500 11px/1.4 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  .link { background: none; border: 0; padding: 0; color: var(--muted); text-decoration: underline; cursor: pointer; font-size: 0.85rem; }
  .link:disabled { opacity: 0.4; cursor: default; }
  .error { color: var(--rose); margin-top: 1rem; }
  code { font: 0.9em var(--mono); background: var(--raised); padding: 0.1em 0.35em; border-radius: 4px; }

  .library { margin-top: 2rem; }
  .library h2 { margin: 0; }
  .library ul { list-style: none; margin: 0.5rem 0 0; padding: 0; display: grid; gap: 0.4rem; }
  .library li { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: 0.65rem 0.9rem;
    background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
  .rec-info { display: grid; min-width: 0; }
  .rec-label { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .rec-meta { color: var(--muted); font: 12px/1.5 var(--mono); }
  .rec-actions { display: flex; align-items: center; gap: 0.75rem; flex: none; }
  footer { margin-top: 2.5rem; color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--line); padding-top: 1rem; }
  footer p { margin: 0.25rem 0; }
</style>
