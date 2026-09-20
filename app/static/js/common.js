/* Shared helpers: health badge, fetch wrapper, formatting. Vanilla JS, no dependencies. */
(function () {
  "use strict";

  var BADGE_CLASSES = {
    joblib_model_v3: "badge--v3",
    joblib_model_v2: "badge--v2",
    joblib_model: "badge--v1",
    mock_rule: "badge--mock"
  };

  function setBadge(el, version, title) {
    if (!el) { return; }
    el.className = "badge " + (BADGE_CLASSES[version] || "badge--error");
    el.textContent = version;
    if (title) { el.title = title; }
  }

  // fetch JSON; resolves {ok, status, body, ms} and never throws on HTTP errors.
  function fetchJSON(url, options) {
    var started = performance.now();
    return fetch(url, options).then(function (res) {
      return res.json().catch(function () { return null; }).then(function (body) {
        return { ok: res.ok, status: res.status, body: body, ms: Math.round(performance.now() - started) };
      });
    });
  }

  function pct(p, digits) {
    return (100 * p).toFixed(digits === undefined ? 1 : digits) + " %";
  }

  function num(v, digits) {
    return typeof v === "number" ? v.toFixed(digits === undefined ? 4 : digits) : "—";
  }

  function fmtInt(n) {
    return typeof n === "number" ? n.toLocaleString("en-GB") : "—";
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "text") { node.textContent = attrs[k]; }
      else if (k === "class") { node.className = attrs[k]; }
      else { node.setAttribute(k, attrs[k]); }
    });
    (children || []).forEach(function (c) { node.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return node;
  }

  // Colour bucket for a probability: a five-step scale, never a binary.
  function riskClass(p) {
    return p < 0.1 ? "r0" : p < 0.3 ? "r1" : p < 0.5 ? "r2" : p < 0.8 ? "r3" : "r4";
  }

  function loadHealth() {
    var badge = document.getElementById("model-badge");
    return fetchJSON("/health", { headers: { Accept: "application/json" } }).then(function (r) {
      if (!r.ok || !r.body) { throw new Error("HTTP " + r.status); }
      setBadge(badge, r.body.model_version, "GET /health: model_version=" + r.body.model_version + (r.body.mock_mode ? " (mock mode)" : ""));
      return r.body;
    }).catch(function (err) {
      setBadge(badge, "unreachable", "GET /health failed: " + err.message);
      return null;
    });
  }

  window.FDE = { setBadge: setBadge, fetchJSON: fetchJSON, pct: pct, num: num, fmtInt: fmtInt, el: el, riskClass: riskClass, health: loadHealth() };
})();
