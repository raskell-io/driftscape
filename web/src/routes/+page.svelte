<script>
  import { onMount } from "svelte";
  import { chooseVoice } from "#lib/voice/voice.js";
  import { LiveEngine } from "#lib/engine.js";
  import * as store from "#lib/store.js";

  let presets = $state({ genres: [], stations: [] });
  let mode = $state("live"); // "live" | "radio"
  let genreId = $state("ambient");
  let prompt = $state("");
  let focus = $state(0); // -2..2 steps of 0.5 guidance
  let wander = $state(0); // -2..2 steps of 0.1 temperature
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

  let resumable = $state(null); // a saved session snapshot
  let recordings = $state([]);
  let storageInfo = $state(null);
  let updateReady = $state(null); // a waiting service worker
  let online = $state(true);

  let engine = null;
  let recordTimer = null;

  const genre = $derived(presets.genres.find((g) => g.id === genreId));
  const station = $derived(presets.stations.find((s) => s.id === stationId));
  const engineReady = $derived(engineState.phase === "ready");
  const waitingSeconds = $derived(startedAt && !nowPlaying ? Math.max(0, (now - startedAt) / 1000) : 0);

  onMount(async () => {
    online = navigator.onLine;
    addEventListener("online", () => (online = true));
    addEventListener("offline", () => (online = false));
    setInterval(() => (now = Date.now()), 1000);
    watchForUpdates();

    const [loadedPresets, settings, session] = await Promise.all([
      fetch("/presets.json").then((r) => r.json()),
      store.loadSettings(),
      store.loadSession(),
    ]);
    presets = loadedPresets;
    if (settings) ({ mode = "live", genreId = "ambient", prompt = "", focus = 0, wander = 0 } = settings);
    settingsLoaded = true;
    if (session?.audio) resumable = session;
    refreshLibrary();

    const voice = await chooseVoice(({ detail, fraction }) => (engineState = { phase: "loading", detail, fraction }));
    voiceLabel = voice.label;
    native = voice.label.includes("engine");
    engine = new LiveEngine(voice, onEvent);
    try {
      await voice.ready();
      engineState = { phase: "ready", detail: "Ready", fraction: 1 };
      // Keep the 570 MB model: ask the browser not to evict our storage.
      if (!native) await store.persist();
    } catch (e) {
      engineState = { phase: "failed", detail: e.message, fraction: 0 };
    }
    refreshLibrary();
  });

  // Remember settings for the next visit.
  $effect(() => {
    const settings = { mode, genreId, prompt, focus, wander };
    if (settingsLoaded) store.saveSettings(settings);
  });

  async function refreshLibrary() {
    recordings = await store.listRecordings().catch(() => []);
    storageInfo = await store.storage().catch(() => null);
  }

  /** A new deploy waits (see service-worker.js); offer it instead of switching under a playing session. */
  async function watchForUpdates() {
    const reg = await navigator.serviceWorker?.getRegistration();
    if (!reg) return;
    const offer = (worker) => {
      if (worker && navigator.serviceWorker.controller) updateReady = worker;
    };
    offer(reg.waiting);
    reg.addEventListener("updatefound", () => {
      const w = reg.installing;
      w?.addEventListener("statechange", () => w.state === "installed" && offer(w));
    });
    // Reload only for an update the listener asked for; on a first visit the new worker also takes control, and that
    // must not reload a page that may already be playing.
    navigator.serviceWorker.addEventListener("controllerchange", () => updateRequested && location.reload());
    setInterval(() => reg.update().catch(() => {}), 30 * 60 * 1000);
  }
  let updateRequested = false;
  function applyUpdate() {
    updateRequested = true;
    updateReady?.postMessage({ type: "skip-waiting" });
  }

  function onEvent(e) {
    if (e.type === "segment") {
      nowPlaying = e.label;
      // Below about 0.3× real time live mode mostly repeats; say why and what helps.
      if (e.speed != null) slow = !native && e.speed < 0.3;
      log = [{ kind: "segment", text: e.label }, ...log].slice(0, 6);
    } else if (e.type === "repeat") {
      // One line for a run of repeats, with a count.
      const count = log[0]?.kind === "repeat" ? log[0].count + 1 : 1;
      const entry = { kind: "repeat", count, text: `Repeating while the next part is made${count > 1 ? ` (×${count})` : ""}` };
      log = log[0]?.kind === "repeat" ? [entry, ...log.slice(1)] : [entry, ...log].slice(0, 6);
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

  async function playLive() {
    begin();
    stationId = null;
    await engine.start({ genre, prompt, focus: focus * 0.5, variety: wander * 0.1 });
  }

  async function tune(id) {
    if (stationId === id && playing) return stop();
    begin();
    stationId = id;
    const s = presets.stations.find((x) => x.id === id);
    await engine.start({ genre: presets.genres.find((g) => g.id === s.genre), station: s });
  }

  async function resume() {
    const snap = resumable;
    const s = presets.stations.find((x) => x.id === snap.stationId) ?? null;
    const g = presets.genres.find((x) => x.id === snap.params.genreId) ?? genre;
    begin();
    stationId = s?.id ?? null;
    mode = s ? "radio" : "live";
    if (!s) {
      genreId = g.id;
      prompt = snap.params.prompt;
    }
    await engine.start({ genre: g, prompt: snap.params.prompt, station: s, focus: snap.params.focus, variety: snap.params.variety, resume: snap });
  }

  function dismissResume() {
    resumable = null;
    store.clearSession();
  }

  function stop() {
    engine?.stop();
    stationId = null;
    startedAt = 0;
  }

  // Steering while it plays (live mode only).
  function steer(change) {
    if (playing && !stationId) engine.tweak(change);
  }
  function pickGenre(id) {
    genreId = id;
    steer({ genre: presets.genres.find((g) => g.id === id) });
  }
  function submitPrompt(event) {
    event.preventDefault();
    if (playing && !stationId) steer({ prompt });
    else if (engineReady) playLive();
  }

  async function toggleRecording() {
    if (!recording) {
      recording = true;
      await engine.startRecording(station?.name ?? `${genre?.name ?? "Live"} · ${prompt.trim() || genre?.default_prompt || ""}`);
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
    const stamp = new Date(rec.startedAt).toISOString().slice(0, 16).replace(/[-:T]/g, "");
    a.download = `driftscape-${stamp}.wav`;
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
    <img class="icon" src="/icon.svg" alt="" width="40" height="40" />
    <h1>
      <img class="wordmark light" src="/wordmark-light.svg" alt="Driftscape" />
      <img class="wordmark dark" src="/wordmark-dark.svg" alt="" aria-hidden="true" />
    </h1>
  </div>
  <p class="engine" class:ready={engineReady} class:failed={engineState.phase === "failed"} role="status">
    {#if engineState.phase === "loading"}
      <span class="dot"></span>{engineState.detail}{engineState.fraction > 0 && engineState.fraction < 1
        ? ` · ${Math.round(engineState.fraction * 100)} %`
        : ""}
    {:else if engineReady}
      <span class="dot"></span>Engine ready
    {:else if engineState.phase === "failed"}
      {engineState.detail}
    {:else}
      <span class="dot"></span>{engineState.detail}
    {/if}
  </p>
</header>

<p class="lede">Endless ambient music, made on your own device while you listen. Describe a mood, or tune in to a station.</p>

{#if engineState.phase === "loading"}
  <div class="loadbar" aria-hidden="true"><span style:--f={engineState.fraction}></span></div>
  <p class="hint">The first visit downloads the model once (about 570 MB). After that it starts in seconds, even offline.</p>
{/if}

{#if updateReady}
  <div class="banner" role="status">
    <span>A new version of Driftscape is ready.</span>
    <button class="secondary small" onclick={applyUpdate}>{playing ? "Reload (stops the music)" : "Reload"}</button>
  </div>
{/if}

{#if !online}
  <div class="banner quiet" role="status">
    {#if storageInfo?.modelCached || native}You're offline. Everything still works: the music is made on this device.
    {:else}You're offline, and the model isn't downloaded yet. Connect once to fetch it; after that Driftscape works offline.{/if}
  </div>
{/if}

{#if resumable && !playing}
  <section class="resume" aria-label="Continue where you left off">
    <div>
      <span class="label">Continue where you left off</span>
      <p class="resume-text">{resumable.label}</p>
      <p class="resume-when">{ago(resumable.savedAt)} · starts at once and carries on from there</p>
    </div>
    <div class="resume-actions">
      <button class="play small" onclick={resume} disabled={!engineReady} aria-label="Continue">▶</button>
      <button class="link" onclick={dismissResume}>Dismiss</button>
    </div>
  </section>
{/if}

<nav class="modes" aria-label="Mode">
  <button class:active={mode === "live"} aria-pressed={mode === "live"} onclick={() => (mode = "live")}>Live</button>
  <button class:active={mode === "radio"} aria-pressed={mode === "radio"} onclick={() => (mode = "radio")}>Radio</button>
  <button disabled title="Coming later: a small timeline to arrange, revisit and export">Producer</button>
</nav>

{#if mode === "live"}
  <section class="panel" aria-labelledby="live-title">
    <h2 id="live-title" class="sr">Live</h2>
    <form class="promptrow" onsubmit={submitPrompt}>
      <label for="prompt" class="label">What should it sound like?</label>
      <div class="field">
        <input
          id="prompt"
          type="text"
          bind:value={prompt}
          placeholder={genre?.default_prompt ?? ""}
          autocomplete="off"
          enterkeyhint="go"
        />
        {#if playing && !stationId}
          <button type="submit" class="secondary">Steer</button>
        {/if}
      </div>
    </form>

    <div class="label">Genre</div>
    <div class="genres" role="radiogroup" aria-label="Genre">
      {#each presets.genres as g (g.id)}
        <button
          role="radio"
          aria-checked={g.id === genreId}
          class="genre"
          class:active={g.id === genreId}
          onclick={() => pickGenre(g.id)}
        >
          <span class="gname">{g.name}</span>
          <span class="gdesc">{g.description}</span>
          {#if g.bpm}<span class="tag">{g.bpm} BPM</span>{/if}
        </button>
      {/each}
    </div>

    <div class="knobs">
      <label>
        <span class="label">Focus <span class="value">{focus > 0 ? "+" : ""}{focus}</span></span>
        <input type="range" min="-2" max="2" step="1" bind:value={focus} onchange={() => steer({ focus: focus * 0.5 })} />
        <span class="ends"><span>loose</span><span>follows the prompt</span></span>
      </label>
      <label>
        <span class="label">Wander <span class="value">{wander > 0 ? "+" : ""}{wander}</span></span>
        <input type="range" min="-2" max="2" step="1" bind:value={wander} onchange={() => steer({ variety: wander * 0.1 })} />
        <span class="ends"><span>steady</span><span>adventurous</span></span>
      </label>
    </div>

    <div class="transport">
      {#if playing && !stationId}
        <button class="play stop" onclick={stop} aria-label="Stop">■</button>
      {:else}
        <button class="play" onclick={playLive} disabled={!engineReady} aria-label="Play">▶</button>
      {/if}
      <p class="transport-hint">
        {#if playing && !stationId}Type a new prompt or pick a genre to steer; you'll hear it within seconds.
        {:else if engineReady}Press play. {native ? "The first part takes a few seconds to make." : "In the browser the first part takes 10–20 seconds to make."}
        {:else}Waiting for the engine.{/if}
      </p>
    </div>
  </section>
{:else}
  <section class="panel" aria-labelledby="radio-title">
    <h2 id="radio-title" class="sr">Radio</h2>
    <div class="stations">
      {#each presets.stations as s (s.id)}
        <button class="station" class:active={s.id === stationId && playing} disabled={!engineReady} onclick={() => tune(s.id)}>
          <span class="sname">{s.name}</span>
          <span class="sdesc">{s.description}</span>
          <span class="tag">{presets.genres.find((g) => g.id === s.genre)?.name}</span>
          <span class="splay" aria-hidden="true">{s.id === stationId && playing ? "■" : "▶"}</span>
        </button>
      {/each}
    </div>
  </section>
{/if}

{#if playing || recording}
  <section class="now" aria-live="polite">
    <div class="now-head">
      <span class="label">Now playing</span>
      <button class="record" class:on={recording} onclick={toggleRecording}>
        {recording ? `● Stop recording · ${clock(recordedSeconds)}` : "● Record"}
      </button>
    </div>
    <p class="nowtext">
      {#if playing}{nowPlaying ||
          `${station ? `Tuning in to ${station.name}` : "Making the first part"}… ${Math.round(progress * 100)} % · ${Math.floor(waitingSeconds)} s`}
      {:else}Stopped. Stop the recording to keep it, or press play to record more.{/if}
    </p>
    {#if slow}
      <p class="hint">This browser makes music slower than it plays, so parts repeat while the next is made. For smooth,
        endless play, install the command-line app and run <code>driftscape serve</code>, which uses your GPU directly.</p>
    {/if}
    <div class="genbar" aria-hidden="true"><span style:--f={progress}></span></div>
    <ol class="log">
      {#each log as entry, i (i)}
        <li class:repeat={entry.kind === "repeat"}>{entry.text}</li>
      {/each}
    </ol>
  </section>
{/if}

{#if error}<p class="error" role="alert">{error}</p>{/if}

{#if recordings.length}
  <section class="library" aria-labelledby="rec-title">
    <h2 id="rec-title" class="label">Your recordings</h2>
    <p class="hint">Kept in this browser until you delete them. They survive reloads and crashes.</p>
    <ul>
      {#each recordings as rec (rec.id)}
        <li>
          <div class="rec-info">
            <span class="rec-label">{rec.label}</span>
            <span class="rec-meta">{when(rec.startedAt)} · {clock(rec.frames / rec.sampleRate)}{rec.done ? "" : recording ? " · recording" : " · interrupted"}</span>
          </div>
          <div class="rec-actions">
            <button class="secondary small" onclick={() => saveRecording(rec)} disabled={!rec.frames}>Save WAV</button>
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
      {storageInfo.modelCached ? "The model is saved on this device" : "The model isn't downloaded yet"}{storageInfo.persisted
        ? " and kept"
        : ""} · {mb(storageInfo.used)} used{#if storageInfo.modelCached && !playing}
        · <button class="link" onclick={removeModel}>Remove the model</button>{/if}
    </p>
  {/if}
</footer>

<style>
  .band { height: 6px; background: var(--dusk); margin: 0 calc(-1 * max(16px, env(safe-area-inset-left, 0px))); }
  .top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; padding-block: 1.5rem 0.5rem; }
  .brand { display: flex; align-items: center; gap: 0.75rem; }
  .icon { border-radius: 10px; }
  h1 { margin: 0; line-height: 0; }
  .wordmark { height: 30px; width: auto; }
  .wordmark.light { display: none; }
  @media (prefers-color-scheme: light) {
    .wordmark.light { display: block; }
    .wordmark.dark { display: none; }
  }
  .engine { margin: 0; font: 500 12px/1.4 var(--mono); letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted);
    display: flex; align-items: center; gap: 0.5rem; }
  .engine.failed { color: var(--rose); text-transform: none; letter-spacing: 0; font-family: var(--body); font-size: 14px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--violet); animation: pulse 1.6s ease-in-out infinite; }
  .engine.ready .dot { background: var(--good); animation: none; }
  @keyframes pulse { 50% { opacity: 0.35; } }
  .lede { color: var(--muted); font-size: 1.05rem; max-width: 60ch; margin: 0.25rem 0 1.25rem; }
  .hint { color: var(--muted); font-size: 0.9rem; margin: 0.5rem 0 1rem; }
  .loadbar, .genbar { height: 4px; background: var(--line); border-radius: 2px; overflow: hidden; }
  .loadbar span, .genbar span { display: block; height: 100%; width: calc(var(--f, 0) * 100%); background: var(--dusk); transition: width 0.3s; }

  .modes { display: flex; gap: 0.25rem; margin: 1.5rem 0 1rem; border-bottom: 1px solid var(--line); }
  .modes button { background: none; border: 0; padding: 0.6rem 1rem; font: 600 0.95rem var(--display); color: var(--muted); cursor: pointer;
    border-bottom: 2px solid transparent; margin-bottom: -1px; }
  .modes button.active { color: var(--ink); border-bottom-color: var(--accent); }
  .modes button:disabled { opacity: 0.45; cursor: default; }

  .panel { display: flex; flex-direction: column; gap: 0.6rem; }
  .label { font: 500 12px/1.4 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin-top: 0.6rem; }
  .value { color: var(--ink); }
  .field { display: flex; gap: 0.5rem; }
  .field input { flex: 1; min-width: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 0.85rem 1rem; font-size: 1.05rem; }
  .field input::placeholder { color: var(--muted); opacity: 0.8; }
  .secondary { background: var(--raised); border: 1px solid var(--line); border-radius: 12px; padding: 0 1.1rem; cursor: pointer; font-weight: 600; }

  .genres { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 0.5rem; }
  .genre { text-align: left; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 0.75rem 0.85rem; cursor: pointer;
    display: flex; flex-direction: column; gap: 0.25rem; min-width: 0; }
  .genre.active { border-color: var(--accent); background: var(--raised); }
  .gname, .sname { font: 600 0.98rem/1.25 var(--display); }
  .gdesc, .sdesc { font-size: 0.84rem; color: var(--muted); line-height: 1.4; }
  .tag { align-self: flex-start; font: 500 11px/1 var(--mono); letter-spacing: 0.04em; color: var(--violet); border: 1px solid var(--line); border-radius: 999px; padding: 0.25rem 0.5rem; margin-top: 0.2rem; }

  .knobs { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-top: 0.4rem; }
  @media (max-width: 560px) { .knobs { grid-template-columns: 1fr; } }
  .knobs label { display: flex; flex-direction: column; gap: 0.2rem; }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  .ends { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted); }

  .transport { display: flex; align-items: center; gap: 1rem; margin-top: 1rem; }
  .play { width: 64px; height: 64px; border-radius: 50%; border: 0; background: var(--accent); color: var(--on-accent); font-size: 1.4rem; cursor: pointer;
    flex: none; box-shadow: 0 6px 24px color-mix(in srgb, var(--accent) 35%, transparent); }
  .play:disabled { opacity: 0.4; cursor: default; box-shadow: none; }
  .play.stop { background: var(--raised); color: var(--ink); box-shadow: none; border: 1px solid var(--line); }
  .transport-hint { margin: 0; color: var(--muted); font-size: 0.92rem; }

  .stations { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 0.6rem; }
  .station { position: relative; text-align: left; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 3rem 1rem 1rem;
    cursor: pointer; display: flex; flex-direction: column; gap: 0.3rem; min-width: 0; }
  .station.active { border-color: var(--accent); background: var(--raised); }
  .station:disabled { opacity: 0.5; cursor: default; }
  .splay { position: absolute; right: 1rem; top: 1rem; width: 32px; height: 32px; border-radius: 50%; display: grid; place-items: center; font-size: 0.8rem;
    background: var(--raised); color: var(--accent); }
  .station.active .splay { background: var(--accent); color: var(--on-accent); }

  .now { margin-top: 1.5rem; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; display: flex; flex-direction: column; gap: 0.5rem; }
  .now-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
  .now-head .label { margin: 0; }
  .nowtext { margin: 0; font: 600 1rem/1.4 var(--display); }
  .record { background: none; border: 1px solid var(--line); border-radius: 999px; padding: 0.3rem 0.8rem; font: 500 12px var(--mono); cursor: pointer; color: var(--muted); }
  .record.on { color: var(--rose); border-color: var(--rose); }
  .log { list-style: none; margin: 0.25rem 0 0; padding: 0; display: flex; flex-direction: column; gap: 0.2rem; font: 12px/1.5 var(--mono); color: var(--muted); }
  .log li:first-child { color: var(--ink); }
  .log li.repeat { color: var(--violet); }

  .error { color: var(--rose); margin-top: 1rem; }
  .banner { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin: 0.75rem 0;
    padding: 0.7rem 1rem; border-radius: 12px; background: var(--raised); border: 1px solid var(--accent); font-size: 0.92rem; }
  .banner.quiet { border-color: var(--line); color: var(--muted); }
  .small { padding: 0.35rem 0.8rem; font-size: 0.85rem; border-radius: 999px; }
  .link { background: none; border: 0; padding: 0; color: var(--muted); text-decoration: underline; cursor: pointer; font-size: 0.85rem; }
  .link:disabled { opacity: 0.4; cursor: default; }
  .resume { display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin: 1rem 0 0.5rem; padding: 1rem 1.1rem;
    background: var(--surface); border: 1px solid var(--line); border-radius: 14px; }
  .resume .label { margin: 0; }
  .resume-text { margin: 0.25rem 0 0; font: 600 1rem/1.4 var(--display); }
  .resume-when { margin: 0.15rem 0 0; color: var(--muted); font-size: 0.85rem; }
  .resume-actions { display: flex; align-items: center; gap: 0.75rem; flex: none; }
  .play.small { width: 44px; height: 44px; font-size: 1rem; padding: 0; }
  .library { margin-top: 2rem; }
  .library h2 { margin: 0; }
  .library ul { list-style: none; margin: 0.5rem 0 0; padding: 0; display: flex; flex-direction: column; gap: 0.4rem; }
  .library li { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: 0.7rem 0.9rem;
    background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
  .rec-info { display: flex; flex-direction: column; min-width: 0; }
  .rec-label { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .rec-meta { color: var(--muted); font: 12px/1.5 var(--mono); }
  .rec-actions { display: flex; align-items: center; gap: 0.75rem; flex: none; }
  code { font: 0.9em var(--mono); background: var(--raised); padding: 0.1em 0.35em; border-radius: 4px; }
  footer { margin-top: 3rem; color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--line); padding-top: 1rem; }
  footer p { margin: 0.25rem 0; }
  .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
</style>
