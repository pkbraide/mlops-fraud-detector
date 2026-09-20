/* Batch page: client-side CSV checks, upload to /v2/batch, results table, CSV download. */
(function () {
  "use strict";
  var F = window.FDE;
  var MAX_BYTES = 2 * 1024 * 1024, MAX_ROWS = 1000;
  var COLS = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price", "repeat_retailer", "used_chip", "used_pin_number", "online_order"];

  var form = document.getElementById("batch-form");
  var fileInput = document.getElementById("csv-file");
  var submitBtn = document.getElementById("batch-submit");
  var downloadBtn = document.getElementById("download-results");
  var lastResult = null;

  function $(id) { return document.getElementById(id); }
  function showPanel(name) {
    ["batch-idle", "batch-loading", "batch-error", "batch-results"].forEach(function (id) { $(id).hidden = id !== "batch-" + name; });
    downloadBtn.hidden = name !== "results";
  }
  function setFileError(msg) { $("file-error").textContent = msg || ""; submitBtn.disabled = !!msg || !fileInput.files.length; }
  function setLoading(on) {
    submitBtn.disabled = on; submitBtn.classList.toggle("is-loading", on);
    submitBtn.querySelector(".btn__label").textContent = on ? "Scoring…" : "Score file";
  }

  // Reject obviously bad files before uploading: extension, size, row count, header.
  function inspectFile(file) {
    $("file-info").textContent = "";
    if (!file) { setFileError(""); return; }
    if (!/\.csv$/i.test(file.name)) { setFileError("Only .csv files are accepted."); return; }
    if (file.size > MAX_BYTES) { setFileError("File is " + (file.size / 1048576).toFixed(1) + " MB; the limit is 2 MB."); return; }
    var reader = new FileReader();
    reader.onload = function () {
      var lines = String(reader.result).split(/\r?\n/).filter(function (l) { return l.trim() !== ""; });
      var rows = Math.max(lines.length - 1, 0);
      var header = (lines[0] || "").split(",").map(function (h) { return h.trim().replace(/^"|"$/g, ""); });
      var missing = COLS.filter(function (c) { return header.indexOf(c) === -1; });
      if (rows === 0) { setFileError("The file has no data rows."); return; }
      if (rows > MAX_ROWS) { setFileError("File has " + rows + " data rows; the limit is " + MAX_ROWS + "."); return; }
      if (missing.length) { setFileError("Missing column(s): " + missing.join(", ")); return; }
      $("file-info").textContent = file.name + " · " + (file.size / 1024).toFixed(1) + " KB · " + rows + " data row" + (rows === 1 ? "" : "s");
      setFileError("");
    };
    reader.onerror = function () { setFileError("Could not read the file."); };
    reader.readAsText(file.slice(0, MAX_BYTES + 1));
  }

  function stat(label, value, extraClass) {
    return F.el("div", { class: "stat" + (extraClass ? " " + extraClass : "") }, [
      F.el("span", { class: "stat__label", text: label }), F.el("span", { class: "stat__value", text: value })
    ]);
  }

  function render(body) {
    lastResult = body;
    var s = body.summary, stats = $("batch-stats");
    stats.innerHTML = "";
    stats.appendChild(stat("Rows scored", F.fmtInt(s.total)));
    stats.appendChild(stat("Flagged", F.fmtInt(s.flagged), s.flagged ? "stat--flag" : ""));
    stats.appendChild(stat("Flag rate", F.pct(s.flag_rate)));
    stats.appendChild(stat("Processing", s.processing_ms + " ms"));
    stats.appendChild(stat("Threshold", F.num(s.threshold, 2)));
    $("batch-served").textContent = "Served by " + body.model_version + (body.rows.length && body.rows[0].confidence === null ? " — rule-based fallback, so no confidence values." : ".");

    var tbody = $("batch-table").querySelector("tbody");
    tbody.innerHTML = "";
    body.rows.forEach(function (r) {
      var i = r.inputs, flagged = r.prediction === 1;
      var tr = F.el("tr", { class: flagged ? "is-flagged" : "" }, [
        F.el("td", { class: "num", text: String(r.row) }),
        F.el("td", {}, [F.el("span", { class: "chip " + (flagged ? "chip--fraud" : "chip--legit"), text: flagged ? "⚠ FRAUD" : "✓ legit" })]),
        F.el("td", { class: "num", text: r.confidence === null ? "—" : F.pct(r.confidence) }),
        F.el("td", { class: "num", text: F.num(i.distance_from_home, 2) }),
        F.el("td", { class: "num", text: F.num(i.distance_from_last_transaction, 2) }),
        F.el("td", { class: "num", text: F.num(i.ratio_to_median_purchase_price, 2) }),
        F.el("td", { text: i.repeat_retailer ? "yes" : "no" }),
        F.el("td", { text: i.used_chip ? "yes" : "no" }),
        F.el("td", { text: i.used_pin_number ? "yes" : "no" }),
        F.el("td", { text: i.online_order ? "yes" : "no" })
      ]);
      tbody.appendChild(tr);
    });
    showPanel("results");
  }

  function fail(message) { $("batch-error-message").textContent = message; showPanel("error"); }

  function submit(e) {
    e.preventDefault();
    var file = fileInput.files[0];
    if (!file) { setFileError("Choose a CSV file first."); return; }
    var thr = Number($("batch-threshold").value);
    if (!(thr > 0 && thr < 1)) { fail("Threshold must be strictly between 0 and 1."); return; }
    var data = new FormData();
    data.append("file", file, file.name);
    setLoading(true); showPanel("loading");
    F.fetchJSON("/v2/batch?threshold=" + encodeURIComponent(thr), { method: "POST", body: data, headers: { Accept: "application/json" } })
      .then(function (r) {
        if (!r.ok) {
          var d = r.body && r.body.detail;
          fail("HTTP " + r.status + ": " + (typeof d === "string" ? d : Array.isArray(d) ? d.map(function (x) { return x.msg; }).join("; ") : "request rejected."));
          return;
        }
        render(r.body);
      })
      .catch(function (err) { fail("Could not reach the service: " + (err.message || "network error") + ". If it was idle, wait a minute and retry."); })
      .then(function () { setLoading(false); });
  }

  function download() {
    if (!lastResult) { return; }
    var head = ["row"].concat(COLS, ["prediction", "is_fraud", "confidence", "threshold", "model_version"]);
    var lines = [head.join(",")];
    lastResult.rows.forEach(function (r) {
      lines.push([r.row].concat(COLS.map(function (c) { return r.inputs[c]; }), [r.prediction, r.is_fraud, r.confidence === null ? "" : r.confidence, lastResult.threshold, lastResult.model_version]).join(","));
    });
    var blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "fraud_scores.csv";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }

  fileInput.addEventListener("change", function () { inspectFile(fileInput.files[0]); });
  form.addEventListener("submit", submit);
  downloadBtn.addEventListener("click", download);
  var zone = document.querySelector(".dropzone");
  zone.addEventListener("dragover", function (e) { e.preventDefault(); });
  zone.addEventListener("drop", function (e) {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; inspectFile(fileInput.files[0]); }
  });
})();
