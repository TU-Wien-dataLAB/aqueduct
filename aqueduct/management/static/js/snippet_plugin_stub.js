// Shared plugin stub builder used by the Snippet admin change form and the
// Plugin test console. Mirrors `gateway.decorators.plugins.generate_plugin_stub`
// so client-side generation matches what the server would produce.
//
// `selected` is an array of {key, type_name, response_kind} objects (the shape
// produced by `get_plugin_endpoint_registry()`).

(function () {
  "use strict";

  // Best-effort sample request bodies per endpoint, used to prefill the test
  // console's "Test input" field so the different per-endpoint requirements are
  // visible. Keys are normalized endpoint paths (no `/v1/`).
  var SAMPLE_PAYLOADS = {
    "chat/completions": '{\n  "model": "gpt-4.1-nano",\n  "messages": [{"role": "user", "content": "Hello"}]\n}',
    "completions": '{\n  "model": "gpt-4.1-nano",\n  "prompt": "Hello"\n}',
    "embeddings": '{\n  "model": "text-embedding-3-small",\n  "input": "Hello"\n}',
    "audio/speech": '{\n  "model": "tts-1",\n  "input": "Hello",\n  "voice": "alloy"\n}',
    "audio/transcriptions": '{\n  "model": "whisper-1",\n  "file": "@audio.mp3"\n}',
    "images/generations": '{\n  "model": "dall-e-3",\n  "prompt": "A cat"\n}',
    "responses": '{\n  "model": "gpt-4.1-nano",\n  "input": "Hello"\n}',
    "decisions": '{\n  "model": "gpt-4.1-nano",\n  "input": "Hello"\n}'
  };

  var DEFAULT_PAYLOAD = '{\n  "model": "gpt-4.1-nano",\n  "messages": [{"role": "user", "content": "Hello"}]\n}';

  function buildPluginStub(selected) {
    var keys = Array.from(new Set(selected.map(function (e) { return e.key; }))).sort();
    var types = Array.from(new Set(selected.map(function (e) { return e.type_name; }).filter(Boolean))).sort();
    var hasJson = selected.some(function (e) {
      return e.response_kind === "json" || e.response_kind === "json_or_streaming";
    });
    var hasStreaming = selected.some(function (e) {
      return e.response_kind === "streaming" || e.response_kind === "json_or_streaming";
    });
    var bodyType = types.length === 1 ? types[0] : "dict";

    var lines = ["from __future__ import annotations", "", "class MyPlugin(PluginSnippet):"];
    if (keys.length) {
      lines.push("    # Runs on: " + keys.join(", "));
      lines.push("    endpoints = [" + keys.map(function (k) { return JSON.stringify(k); }).join(", ") + "]");
    } else {
      lines.push("    # No endpoints selected: the plugin runs on every endpoint.");
      lines.push("    endpoints = []");
    }
    lines.push("");
    lines.push("    def before_request(self, request, token, body: " + bodyType + ") -> dict | None:");
    if (types.length) {
      lines.push("        # Request body is validated against: " + types.join(", "));
    }
    lines.push('        """Optionally transform/replace the request body.');
    lines.push("");
    lines.push("        Return the (possibly mutated) body to pass downstream, or ``None`` to");
    lines.push("        leave the body unchanged. Raise ``BlockedByPlugin`` to stop the request.");
    lines.push('        """');
    lines.push("        return body");
    lines.push("");
    if (hasJson) {
      lines.push("    def after_response(self, response) -> None:");
      lines.push('        """Non-streaming only. Return a replacement response, or ``None`` to keep unchanged."""');
      lines.push("        return None");
      lines.push("");
    }
    if (hasStreaming) {
      lines.push("    def after_streaming_response(self, response) -> None:");
      lines.push('        """Streaming wrapper only (headers/status, not content)."""');
      lines.push("        return None");
    }
    return lines.join("\n") + "\n";
  }

  function getSamplePayload(selected) {
    var key = selected && selected.length ? selected[0].key : null;
    return SAMPLE_PAYLOADS[key] || DEFAULT_PAYLOAD;
  }

  window.buildPluginStub = buildPluginStub;
  window.getSamplePayload = getSamplePayload;
})();
