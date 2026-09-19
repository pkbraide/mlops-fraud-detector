/* Enterprise Real-Time Fraud Detection Engine — page logic.
   Vanilla JS, no dependencies. Talks to GET /health and POST /v2/predict. */

(function () {
  "use strict";

  var NUMERIC_FIELDS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price"
  ];
  var BINARY_FIELDS = ["repeat_retailer", "used_chip", "used_pin_number", "online_order"];
  var ALL_FIELDS = NUMERIC_FIELDS.concat(BINARY_FIELDS);

  // Both examples are classified the same way by the trained v2 model and by
  // the rule-based fallback, so the buttons behave identically in either mode.
  var EXAMPLES = {
    legit: {
      distance_from_home: 5.2,
      distance_from_last_transaction: 0.4,
      ratio_to_median_purchase_price: 0.9,
      repeat_retailer: 1,
      used_chip: 1,
      used_pin_number: 0,
      online_order: 0
    },
    fraud: {
      distance_from_home: 210.5,
      distance_from_last_transaction: 12.3,
      ratio_to_median_purchase_price: 6.2,
      repeat_retailer: 0,
      used_chip: 0,
      used_pin_number: 0,
      online_order: 1
    }
  };

  var BADGE_CLASSES = {
    joblib_model_v2: "badge--v2",
    joblib_model: "badge--v1",
    mock_rule: "badge--mock"
  };

  var form = document.getElementById("predict-form");
  var submitBtn = document.getElementById("submit-btn");
  var submitLabel = submitBtn.querySelector(".btn__label");
  var badge = document.getElementById("model-badge");

  var panels = {
    idle: document.getElementById("result-idle"),
    loading: document.getElementById("result-loading"),
    verdict: document.getElementById("result-verdict"),
    error: document.getElementById("result-error")
  };
  var verdictEl = document.getElementById("result-verdict");
  var verdictLabel = document.getElementById("verdict-label");
  var verdictSub = document.getElementById("verdict-sub");
  var verdictModel = document.getElementById("verdict-model");
  var verdictLatency = document.getElementById("verdict-latency");
  var verdictRaw = document.getElementById("verdict-raw");
  var errorMessage = document.getElementById("error-message");
  var errorList = document.getElementById("error-list");

  // ------------------------------------------------------------ helpers
  function field(name) {
    return form.elements[name];
  }

  function showPanel(name) {
    Object.keys(panels).forEach(function (key) {
      panels[key].hidden = key !== name;
    });
  }

  function setBadge(version, title) {
    badge.className = "badge " + (BADGE_CLASSES[version] || "badge--error");
    badge.textContent = version;
    if (title) {
      badge.title = title;
    }
  }

  function setFieldError(name, message) {
    var input = field(name);
    var target = form.querySelector('[data-error-for="' + name + '"]');
    if (input) {
      if (message) {
        input.setAttribute("aria-invalid", "true");
      } else {
        input.removeAttribute("aria-invalid");
      }
    }
    if (target) {
      target.textContent = message || "";
    }
  }

  function clearFieldErrors() {
    ALL_FIELDS.forEach(function (name) {
      setFieldError(name, "");
    });
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.classList.toggle("is-loading", isLoading);
    submitLabel.textContent = isLoading ? "Scoring…" : "Score transaction";
    Array.prototype.forEach.call(form.querySelectorAll("[data-example]"), function (btn) {
      btn.disabled = isLoading;
    });
  }

  function humaniseField(name) {
    return name.replace(/_/g, " ");
  }

  // ------------------------------------------------------------ validation
  // Mirrors the Pydantic constraints on FraudRequestV2:
  //   distances / ratio: float, ge=0        binary fields: int, ge=0, le=1
  function readAndValidate() {
    clearFieldErrors();
    var payload = {};
    var valid = true;

    NUMERIC_FIELDS.forEach(function (name) {
      var raw = field(name).value.trim();
      var value = raw === "" ? NaN : Number(raw);
      if (raw === "") {
        setFieldError(name, "Required.");
        valid = false;
      } else if (!isFinite(value)) {
        setFieldError(name, "Must be a number.");
        valid = false;
      } else if (value < 0) {
        setFieldError(name, "Must be 0 or greater.");
        valid = false;
      } else {
        payload[name] = value;
      }
    });

    BINARY_FIELDS.forEach(function (name) {
      var value = Number(field(name).value);
      if (value !== 0 && value !== 1) {
        setFieldError(name, "Must be 0 or 1.");
        valid = false;
      } else {
        payload[name] = value;
      }
    });

    return valid ? payload : null;
  }

  // ------------------------------------------------------------ examples
  function loadExample(key) {
    var example = EXAMPLES[key];
    if (!example) {
      return;
    }
    ALL_FIELDS.forEach(function (name) {
      field(name).value = String(example[name]);
    });
    clearFieldErrors();
    showPanel("idle");
    field(NUMERIC_FIELDS[0]).focus({ preventScroll: true });
  }

  // ------------------------------------------------------------ rendering
  function renderVerdict(result, latencyMs) {
    var isFraud = result.prediction === 1 || result.is_fraud === true;
    verdictEl.classList.remove("verdict--fraud", "verdict--legit");
    verdictEl.classList.add(isFraud ? "verdict--fraud" : "verdict--legit");
    verdictLabel.textContent = isFraud ? "FRAUD" : "LEGITIMATE";
    verdictSub.textContent = isFraud
      ? "This transaction matches patterns the model associates with fraud."
      : "This transaction looks consistent with the cardholder's normal behaviour.";
    verdictModel.textContent = result.model_version || "unknown";
    verdictLatency.textContent = latencyMs + " ms round trip";
    verdictRaw.textContent = JSON.stringify(result);
    showPanel("verdict");
  }

  function renderError(message, details) {
    errorMessage.textContent = message;
    errorList.innerHTML = "";
    (details || []).forEach(function (line) {
      var li = document.createElement("li");
      li.textContent = line;
      errorList.appendChild(li);
    });
    showPanel("error");
  }

  // Turn a FastAPI 422 body into per-field messages plus a summary list.
  function applyValidationErrors(body) {
    var lines = [];
    var detail = body && Array.isArray(body.detail) ? body.detail : [];
    detail.forEach(function (err) {
      var loc = Array.isArray(err.loc) ? err.loc : [];
      var name = loc.length ? String(loc[loc.length - 1]) : "";
      var msg = err.msg || "Invalid value.";
      if (ALL_FIELDS.indexOf(name) !== -1) {
        setFieldError(name, msg);
        lines.push(humaniseField(name) + ": " + msg);
      } else {
        lines.push(msg);
      }
    });
    return lines;
  }

  // ------------------------------------------------------------ network
  function fetchHealth() {
    fetch("/health", { headers: { Accept: "application/json" } })
      .then(function (res) {
        if (!res.ok) {
          throw new Error("HTTP " + res.status);
        }
        return res.json();
      })
      .then(function (body) {
        var version = body.model_version || "unknown";
        var v2 = (body.endpoints || []).filter(function (e) {
          return e.path === "/v2/predict";
        })[0];
        var served = v2 && v2.model_version ? v2.model_version : version;
        setBadge(served, "GET /health reports model_version=" + version + (body.mock_mode ? " (mock mode)" : ""));
      })
      .catch(function (err) {
        setBadge("unreachable", "GET /health failed: " + err.message);
      });
  }

  function submit(event) {
    event.preventDefault();
    var payload = readAndValidate();
    if (!payload) {
      renderError("Please fix the highlighted fields.", []);
      return;
    }

    setLoading(true);
    showPanel("loading");
    var started = performance.now();

    fetch("/v2/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (res) {
        var latency = Math.round(performance.now() - started);
        return res
          .json()
          .catch(function () {
            return null;
          })
          .then(function (body) {
            if (res.status === 422) {
              var lines = applyValidationErrors(body);
              renderError("The API rejected the payload (422).", lines);
              return;
            }
            if (!res.ok) {
              var detail = body && body.detail ? String(body.detail) : "";
              renderError("The API returned HTTP " + res.status + ".", detail ? [detail] : []);
              return;
            }
            if (!body || typeof body.prediction === "undefined") {
              renderError("The API returned an unexpected response.", [JSON.stringify(body)]);
              return;
            }
            renderVerdict(body, latency);
          });
      })
      .catch(function (err) {
        renderError(
          "Could not reach the scoring service.",
          [
            err && err.message ? err.message : "Network error.",
            "If the service was idle, wait a minute for the cold start and try again."
          ]
        );
      })
      .then(function () {
        setLoading(false);
      });
  }

  // ------------------------------------------------------------ wiring
  form.addEventListener("submit", submit);
  Array.prototype.forEach.call(document.querySelectorAll("[data-example]"), function (btn) {
    btn.addEventListener("click", function () {
      loadExample(btn.getAttribute("data-example"));
    });
  });
  ALL_FIELDS.forEach(function (name) {
    field(name).addEventListener("input", function () {
      setFieldError(name, "");
    });
  });

  fetchHealth();
})();
