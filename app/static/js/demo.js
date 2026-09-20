/* Landing page: model selector, v2 form, v3 presets, verdict card, live threshold re-scoring. */
(function () {
  "use strict";
  var F = window.FDE;

  var NUMERIC = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price"];
  var BINARY = ["repeat_retailer", "used_chip", "used_pin_number", "online_order"];
  var ALL = NUMERIC.concat(BINARY);

  // Both v2 examples are classified the same way by the trained model and by the fallback rule.
  var EXAMPLES = {
    legit: { distance_from_home: 5.2, distance_from_last_transaction: 0.4, ratio_to_median_purchase_price: 0.9, repeat_retailer: 1, used_chip: 1, used_pin_number: 0, online_order: 0 },
    fraud: { distance_from_home: 210.5, distance_from_last_transaction: 12.3, ratio_to_median_purchase_price: 6.2, repeat_retailer: 0, used_chip: 0, used_pin_number: 0, online_order: 1 }
  };

  var form = document.getElementById("predict-form");
  var submitBtn = document.getElementById("submit-btn");
  var slider = document.getElementById("threshold");
  var sliderOut = document.getElementById("threshold-out");
  var panels = { idle: "result-idle", loading: "result-loading", verdict: "result-verdict", error: "result-error" };
  var presetsData = JSON.parse(document.getElementById("v3-presets").textContent || "{}");
  var presets = presetsData.presets || [];
  var presetById = {};
  presets.forEach(function (p) { presetById[p.id] = p; });

  var state = { model: "v2", last: null, timer: null };

  function $(id) { return document.getElementById(id); }
  function field(name) { return form.elements[name]; }
  function toggleAttr(node, name, on) { if (on) { node.setAttribute(name, ""); } else { node.removeAttribute(name); } }
  function threshold() { return Number(slider.value); }

  function showPanel(name) {
    Object.keys(panels).forEach(function (k) { $(panels[k]).hidden = k !== name; });
  }

  function setFieldError(name, message) {
    var input = field(name);
    var target = form.querySelector('[data-error-for="' + name + '"]');
    if (input) { if (message) { input.setAttribute("aria-invalid", "true"); } else { input.removeAttribute("aria-invalid"); } }
    if (target) { target.textContent = message || ""; }
  }

  function clearFieldErrors() { ALL.forEach(function (n) { setFieldError(n, ""); }); }

  function setLoading(on) {
    submitBtn.disabled = on;
    submitBtn.classList.toggle("is-loading", on);
    submitBtn.querySelector(".btn__label").textContent = on ? "Scoring…" : "Score transaction";
    Array.prototype.forEach.call(document.querySelectorAll("[data-example], .preset"), function (b) { b.disabled = on; });
  }

  // Mirrors FraudRequestV2: distances/ratio float >= 0, binaries in {0, 1}.
  function readAndValidate() {
    clearFieldErrors();
    var payload = {}, valid = true;
    NUMERIC.forEach(function (name) {
      var raw = field(name).value.trim(), value = raw === "" ? NaN : Number(raw);
      if (raw === "") { setFieldError(name, "Required."); valid = false; }
      else if (!isFinite(value)) { setFieldError(name, "Must be a number."); valid = false; }
      else if (value < 0) { setFieldError(name, "Must be 0 or greater."); valid = false; }
      else { payload[name] = value; }
    });
    BINARY.forEach(function (name) {
      var value = Number(field(name).value);
      if (value !== 0 && value !== 1) { setFieldError(name, "Must be 0 or 1."); valid = false; }
      else { payload[name] = value; }
    });
    return valid ? payload : null;
  }

  function loadExample(key) {
    var ex = EXAMPLES[key];
    if (!ex) { return; }
    ALL.forEach(function (n) { field(n).value = String(ex[n]); });
    clearFieldErrors();
    showPanel("idle");
    state.last = null;
  }

  // ------------------------------------------------------------ rendering
  function renderVerdict(body, ms, meta) {
    var isFraud = body.prediction === 1;
    var box = $("result-verdict");
    box.classList.remove("verdict--fraud", "verdict--legit");
    box.classList.add(isFraud ? "verdict--fraud" : "verdict--legit");
    // SVG elements have no `hidden` IDL property, so toggle the attribute directly.
    toggleAttr($("verdict-icon-fraud"), "hidden", !isFraud);
    toggleAttr($("verdict-icon-legit"), "hidden", isFraud);
    $("verdict-label").textContent = isFraud ? "FRAUD" : "LEGITIMATE";
    $("verdict-sub").textContent = isFraud
      ? "Flagged: probability is at or above the threshold."
      : "Not flagged: probability is below the threshold.";

    var conf = body.confidence, thr = body.threshold;
    var hasConf = typeof conf === "number";
    $("meter-block").hidden = !hasConf;
    $("meter-note").hidden = hasConf;
    if (hasConf) {
      var fill = $("meter-fill");
      fill.style.width = (100 * conf).toFixed(1) + "%";
      fill.className = "meter__fill " + F.riskClass(conf);
      $("meter").setAttribute("aria-valuenow", (100 * conf).toFixed(1));
      $("meter-value").textContent = F.pct(conf) + " · " + F.num(conf);
      $("meter-mark").style.left = (100 * thr).toFixed(1) + "%";
      $("meter-thr").textContent = "threshold " + F.pct(thr, 0);
    } else {
      $("meter-note").textContent = "No probability: the rule-based fallback served this request (model_version = mock_rule), so the threshold has no effect. A real number is never invented here.";
    }

    $("verdict-model").textContent = body.model_version || "unknown";
    $("verdict-threshold").textContent = F.num(thr, 2);
    $("verdict-latency").textContent = ms + " ms round trip";
    $("verdict-raw").textContent = JSON.stringify(body);

    var truthRow = $("truth-row");
    if (meta && meta.preset) {
      truthRow.hidden = false;
      var truth = meta.preset.ground_truth === 1;
      $("verdict-truth").textContent = (truth ? "fraud" : "legitimate") + (truth === isFraud ? " — model agrees" : " — model disagrees at this threshold");
    } else {
      truthRow.hidden = true;
    }
    showPanel("verdict");
  }

  function renderError(message, lines) {
    $("error-message").textContent = message;
    var list = $("error-list");
    list.innerHTML = "";
    (lines || []).forEach(function (t) { list.appendChild(F.el("li", { text: t })); });
    showPanel("error");
  }

  function validationLines(body) {
    var lines = [];
    ((body && body.detail) || []).forEach(function (err) {
      var loc = Array.isArray(err.loc) ? err.loc : [], name = loc.length ? String(loc[loc.length - 1]) : "";
      var msg = err.msg || "Invalid value.";
      if (ALL.indexOf(name) !== -1) { setFieldError(name, msg); lines.push(name.replace(/_/g, " ") + ": " + msg); }
      else { lines.push((name ? name + ": " : "") + msg); }
    });
    return lines;
  }

  // ------------------------------------------------------------ scoring
  function score(url, payload, meta) {
    state.last = { url: url, payload: payload, meta: meta || null };
    setLoading(true);
    showPanel("loading");
    var body = Object.assign({}, payload, { threshold: threshold() });
    F.fetchJSON(url, { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: JSON.stringify(body) })
      .then(function (r) {
        if (r.status === 422) { renderError("The API rejected the payload (422).", validationLines(r.body)); return; }
        if (!r.ok) { renderError("The API returned HTTP " + r.status + ".", r.body && r.body.detail ? [String(r.body.detail)] : []); return; }
        if (!r.body || typeof r.body.prediction === "undefined") { renderError("Unexpected response.", [JSON.stringify(r.body)]); return; }
        renderVerdict(r.body, r.ms, meta);
      })
      .catch(function (err) {
        renderError("Could not reach the scoring service.", [err && err.message ? err.message : "Network error.", "If the service was idle, wait a minute for the cold start and try again."]);
      })
      .then(function () { setLoading(false); });
  }

  function rescore() {
    if (state.last) { score(state.last.url, state.last.payload, state.last.meta); }
  }

  // ------------------------------------------------------------ model selector
  function selectModel(model) {
    state.model = model;
    $("panel-v2").hidden = model !== "v2";
    $("panel-v3").hidden = model !== "v3";
    state.last = null;
    showPanel("idle");
  }

  function renderPresets() {
    var host = $("presets");
    if (!host) { return; }
    presets.forEach(function (p) {
      var chip = F.el("span", { class: "chip " + (p.ground_truth === 1 ? "chip--fraud" : "chip--legit"), text: p.ground_truth === 1 ? "ground truth: fraud" : "ground truth: legitimate" });
      var btn = F.el("button", { type: "button", class: "preset", "data-preset": p.id }, [
        F.el("span", { class: "preset__title" }, [p.title, chip]),
        F.el("span", { class: "preset__desc", text: p.description }),
        F.el("span", { class: "preset__meta", text: "Amount " + p.amount.toFixed(2) + " · hold-out probability " + p.holdout_probability.toFixed(2) + " · " + presetsData.features.length + " features" })
      ]);
      btn.addEventListener("click", function () { score("/v3/predict", p.values, { preset: p }); });
      host.appendChild(btn);
    });
  }

  function applyHealth(health) {
    var models = (health && health.models) || {};
    ["v2", "v3"].forEach(function (v) {
      var node = document.querySelector('[data-model-status="' + v + '"]');
      var info = models[v];
      if (!node) { return; }
      if (!info) { node.textContent = "status unknown"; return; }
      node.textContent = info.loaded ? "artifact loaded · " + info.model_version : "artifact missing · fallback rule (no confidence)";
    });
  }

  // ------------------------------------------------------------ wiring
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var payload = readAndValidate();
    if (!payload) { renderError("Please fix the highlighted fields.", []); return; }
    score("/v2/predict", payload, null);
  });
  Array.prototype.forEach.call(document.querySelectorAll("[data-example]"), function (b) {
    b.addEventListener("click", function () { loadExample(b.getAttribute("data-example")); });
  });
  ALL.forEach(function (n) { field(n).addEventListener("input", function () { setFieldError(n, ""); }); });
  Array.prototype.forEach.call(document.querySelectorAll('input[name="model"]'), function (r) {
    r.addEventListener("change", function () { if (r.checked) { selectModel(r.value); } });
  });
  slider.addEventListener("input", function () {
    sliderOut.value = Number(slider.value).toFixed(2);
    $("meter-mark").style.left = (100 * threshold()).toFixed(1) + "%";
    $("meter-thr").textContent = "threshold " + F.pct(threshold(), 0);
    clearTimeout(state.timer);
    state.timer = setTimeout(rescore, 250);
  });

  renderPresets();
  F.health.then(applyHealth);
})();
