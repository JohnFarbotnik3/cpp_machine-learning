<script lang="ts">
	import "../styles.css";

	const { children } = $props();
</script>

<nav id="nav">
	<div>abc</div>
	<button>def</button>
</nav>
<div id="body">
	{@render children()}
</div>
