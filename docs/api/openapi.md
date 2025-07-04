---
title: List Models
parent: API Reference
nav_order: 4
---

<!-- client-rendered openapi UI copied from FastAPI -->

<link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@3/swagger-ui.css">
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.1/swagger-ui-bundle.js"></script>
<!-- `SwaggerUIBundle` is now available on the page -->

<!-- render the ui here -->
<div id="openapi-ui"></div>

<script>
const ui = SwaggerUIBundle({
  url: 'https://raw.githubusercontent.com/TU-Wien-dataLAB/aqueduct/refs/heads/9-add-api-documentation/docs/openapi.yaml',
  dom_id: '#openapi-ui',
  presets: [
    SwaggerUIBundle.presets.apis,
    SwaggerUIBundle.SwaggerUIStandalonePreset
  ],
  layout: "BaseLayout",
  deepLinking: true,
  showExtensions: true,
  showCommonExtensions: true,
});
</script>