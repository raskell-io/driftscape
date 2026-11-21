<script>
  import "@fontsource-variable/inter";
  import "@fontsource-variable/sora";
  import "@fontsource-variable/jetbrains-mono";
  import "../app.css";
  let { children } = $props();
</script>

<svelte:head><title>Driftscape</title></svelte:head>
<a class="skip" href="#main">Skip to content</a>
<main id="main">{@render children()}</main>
