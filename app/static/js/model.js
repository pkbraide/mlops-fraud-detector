/* Model card: renders everything from GET /metrics. No numbers live in the template. */
(function () {
  "use strict";
  var F = window.FDE;
  var ORDER = ["v3", "v2"];
  var LABELS = { v3: "v3 — ULB creditcard (real)", v2: "v2 — card_transdata (synthetic)", v1: "v1 — synthetic (deprecated)" };

  function $(id) { return document.getElementById(id); }
  function get(obj, path) {
    return path.split(".").reduce(function (o, k) { return o && o[k] !== undefined ? o[k] : undefined; }, obj);
  }

  var ROWS = [
    ["Dataset", function (m) { var d = m.dataset || {}; return d.openml_name ? d.openml_name + " (OpenML " + d.openml_data_id + ")" : "—"; }],
    ["Nature", function (m) { return get(m, "dataset.description") || "—"; }],
    ["Licence", function (m) { return get(m, "dataset.licence") || "—"; }],
    ["Rows", function (m) { return F.fmtInt(get(m, "dataset.n_rows_available")); }],
    ["Fraud prevalence", function (m) { var p = get(m, "dataset.fraud_prevalence"); return typeof p === "number" ? F.pct(p, 3) + " (" + F.fmtInt(get(m, "dataset.fraud_count")) + " frauds)" : "—"; }],
    ["Features", function (m) { return m.n_features + (get(m, "dataset.features_note") ? " — " + m.dataset.features_note : ""); }],
    ["Training rows", function (m) { return F.fmtInt(get(m, "metrics.n_train_rows")); }],
    ["Hold-out rows (frauds)", function (m) { return F.fmtInt(get(m, "metrics.n_test_rows")) + " (" + F.fmtInt(get(m, "metrics.n_test_fraud")) + ")"; }],
    ["Precision (fraud)", function (m) { return F.num(get(m, "metrics.fraud_precision")); }],
    ["Recall (fraud)", function (m) { return F.num(get(m, "metrics.fraud_recall")); }],
    ["F1 (fraud)", function (m) { return F.num(get(m, "metrics.fraud_f1")); }],
    ["PR-AUC", function (m) { return F.num(get(m, "metrics.pr_auc")); }],
    ["Estimator", function (m) { var p = m.model_params || {}; return (m.estimator || "—") + (p.n_estimators ? " · " + p.n_estimators + " trees · class_weight=" + p.class_weight : ""); }],
    ["scikit-learn", function (m) { return m.sklearn_version || "—"; }],
    ["Trained", function (m) { return m.trained_at ? m.trained_at.replace("T", " ").replace("+00:00", " UTC") : "—"; }]
  ];

  function renderTable(models) {
    var body = $("compare-body");
    body.innerHTML = "";
    ROWS.forEach(function (row) {
      var tr = F.el("tr", {}, [F.el("th", { scope: "row", text: row[0] })]);
      ORDER.forEach(function (v) {
        var m = models[v];
        tr.appendChild(F.el("td", { text: m && m.loaded ? row[1](m) : "artifact not loaded" }));
      });
      body.appendChild(tr);
    });
  }

  function cell(value, label, cls) {
    return F.el("div", { class: "cell " + cls }, [F.el("strong", { text: F.fmtInt(value) }), F.el("span", { text: label })]);
  }

  function renderMatrices(models) {
    var host = $("matrices");
    host.innerHTML = "";
    ORDER.forEach(function (v) {
      var m = models[v], cm = get(m, "metrics.confusion_matrix");
      var block = F.el("div", {}, [F.el("h3", { text: LABELS[v] })]);
      if (!cm) { block.appendChild(F.el("p", { class: "muted small", text: "Artifact not loaded on this instance." })); host.appendChild(block); return; }
      var grid = F.el("div", { class: "matrix", role: "table", "aria-label": "Confusion matrix for " + v }, [
        F.el("div", { class: "axis", text: "" }), F.el("div", { class: "axis", text: "predicted legit" }), F.el("div", { class: "axis", text: "predicted fraud" }),
        F.el("div", { class: "axis", text: "actual legit" }), cell(cm.tn, "true negatives", "cell--hit"), cell(cm.fp, "false positives", "cell--miss"),
        F.el("div", { class: "axis", text: "actual fraud" }), cell(cm.fn, "false negatives", "cell--miss"), cell(cm.tp, "true positives", "cell--hit")
      ]);
      block.appendChild(grid);
      var total = cm.tn + cm.fp + cm.fn + cm.tp;
      block.appendChild(F.el("p", { class: "hint", text: F.fmtInt(total) + " hold-out rows · " + F.fmtInt(cm.fn) + " frauds missed · " + F.fmtInt(cm.fp) + " false alarms" }));
      host.appendChild(block);
    });
  }

  function renderImportances(models) {
    var host = $("importances");
    host.innerHTML = "";
    ORDER.forEach(function (v) {
      var m = models[v], imp = get(m, "metrics.feature_importances");
      var block = F.el("div", {}, [F.el("h3", { text: LABELS[v] })]);
      if (!imp) { block.appendChild(F.el("p", { class: "muted small", text: "Artifact not loaded on this instance." })); host.appendChild(block); return; }
      var entries = Object.keys(imp).map(function (k) { return [k, imp[k]]; }).sort(function (a, b) { return b[1] - a[1]; });
      var top = entries.slice(0, 10), rest = entries.slice(10);
      var max = top.length ? top[0][1] : 1;
      var bars = F.el("div", { class: "bars", role: "list" });
      top.forEach(function (e) {
        var fill = F.el("div", { class: "bar__fill" });
        fill.style.width = (100 * e[1] / max).toFixed(1) + "%";
        bars.appendChild(F.el("div", { class: "bar", role: "listitem" }, [
          F.el("span", { class: "bar__name", text: e[0], title: e[0] }),
          F.el("div", { class: "bar__track", "aria-hidden": "true" }, [fill]),
          F.el("span", { class: "bar__value", text: F.num(e[1], 3) })
        ]));
      });
      block.appendChild(bars);
      if (rest.length) {
        var restSum = rest.reduce(function (s, e) { return s + e[1]; }, 0);
        block.appendChild(F.el("p", { class: "hint", text: rest.length + " further features share the remaining " + F.num(restSum, 3) + "." }));
      }
      host.appendChild(block);
    });
  }

  function renderStatus(models) {
    var host = $("artifact-status");
    host.innerHTML = "";
    ["v3", "v2", "v1"].forEach(function (v) {
      var m = models[v];
      if (!m) { return; }
      var line = F.el("div", { class: "status-line" }, [
        F.el("strong", { text: LABELS[v] }),
        F.el("code", { text: m.route }),
        F.el("span", { class: "badge " + (m.loaded ? (v === "v1" ? "badge--v1" : "badge--" + v) : "badge--mock"), text: m.model_version })
      ]);
      host.appendChild(line);
      host.appendChild(F.el("p", { class: "hint", text: (m.loaded ? "Loaded from " + m.artifact : "Not found: " + m.artifact + " — fallback rule active: " + m.fallback_rule) + (m.note ? " " + m.note : "") }));
    });
  }

  F.fetchJSON("/metrics", { headers: { Accept: "application/json" } }).then(function (r) {
    if (!r.ok || !r.body || !r.body.models) { throw new Error("HTTP " + r.status); }
    var models = r.body.models;
    $("metrics-source").textContent = "source: " + r.body.source;
    renderTable(models); renderMatrices(models); renderImportances(models); renderStatus(models);
  }).catch(function (err) {
    $("metrics-error").hidden = false;
    $("metrics-error-message").textContent = err.message + ". Reload the page; if the service was idle it may still be starting.";
    $("compare-body").innerHTML = "";
  });
})();
