<script>
  // A rotary control: drag up or down (or use the arrow keys) to turn. `onchange` fires when the value settles.
  let { label, min = 0, max = 10, step = 1, value = $bindable(0), format = (v) => String(v), onchange } = $props();

  const t = $derived((value - min) / (max - min));
  let start = null;
  let dial;
  // Set through the DOM, not a style attribute: the page's Content Security Policy blocks inline style attributes.
  $effect(() => dial?.style.setProperty("--t", String(t)));

  const snap = (v) => Math.min(max, Math.max(min, Math.round(v / step) * step));
  function down(e) {
    e.currentTarget.setPointerCapture(e.pointerId);
    start = { y: e.clientY, value };
  }
  function move(e) {
    if (!start) return;
    value = snap(start.value + ((start.y - e.clientY) / 140) * (max - min));
  }
  function up() {
    if (!start) return;
    if (value !== start.value) onchange?.(value);
    start = null;
  }
  function key(e) {
    const d = { ArrowUp: step, ArrowRight: step, ArrowDown: -step, ArrowLeft: -step, PageUp: step * 2, PageDown: -step * 2 }[e.key];
    if (d === undefined && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    const before = value;
    value = e.key === "Home" ? min : e.key === "End" ? max : snap(value + d);
    if (value !== before) onchange?.(value);
  }
</script>

<div class="knob">
  <div
    bind:this={dial}
    class="dial"
    role="slider"
    tabindex="0"
    aria-label={label}
    aria-valuemin={min}
    aria-valuemax={max}
    aria-valuenow={value}
    aria-valuetext={format(value)}
    onpointerdown={down}
    onpointermove={move}
    onpointerup={up}
    onpointercancel={up}
    onkeydown={key}
  >
    <span class="pointer"></span>
  </div>
  <span class="name">{label}</span>
  <span class="value">{format(value)}</span>
</div>

<style>
  .knob { display: grid; justify-items: center; gap: 6px; user-select: none; }
  .dial { --size: 58px; width: var(--size); height: var(--size); border-radius: 50%; position: relative; cursor: ns-resize; touch-action: none;
    background: radial-gradient(circle at 40% 32%, #3d355f, #221d3c 65%); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5), 0 0 0 1px #2d2650; }
  /* the value arc, from 7 o'clock round to 5 o'clock */
  .dial::before { content: ""; position: absolute; inset: -7px; border-radius: 50%;
    background: conic-gradient(from 225deg, var(--peach) 0deg, var(--rose) calc(var(--t) * 270deg), #2d2650 calc(var(--t) * 270deg), #2d2650 270deg, transparent 270deg);
    -webkit-mask: radial-gradient(circle, transparent calc(var(--size) / 2 + 3px), #000 calc(var(--size) / 2 + 4px));
    mask: radial-gradient(circle, transparent calc(var(--size) / 2 + 3px), #000 calc(var(--size) / 2 + 4px)); }
  .pointer { position: absolute; left: 50%; top: 6px; width: 3px; height: 15px; margin-left: -1.5px; border-radius: 2px; background: var(--apricot);
    transform-origin: 1.5px calc(var(--size) / 2 - 6px); transform: rotate(calc(-135deg + var(--t) * 270deg)); }
  .dial:focus-visible { outline: 2px solid var(--accent); outline-offset: 9px; }
  .name { font: 600 10px/1 var(--display); letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); }
  .value { font: 500 11px var(--mono); color: var(--apricot); min-width: 6ch; text-align: center; }
</style>
