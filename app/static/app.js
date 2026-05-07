const healthPill = document.getElementById("health-pill");
const analyzeForm = document.getElementById("analyze-form");
const analyzeButton = document.getElementById("analyze-button");
const analyzeError = document.getElementById("analyze-error");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const rootCause = document.getElementById("root-cause");
const severityPill = document.getElementById("severity-pill");
const confidencePill = document.getElementById("confidence-pill");
const suggestedFix = document.getElementById("suggested-fix");
const similarList = document.getElementById("similar-list");
const incidentList = document.getElementById("incident-list");
const uploadForm = document.getElementById("upload-form");
const uploadButton = document.getElementById("upload-button");
const uploadMessage = document.getElementById("upload-message");
const similarRefresh = document.getElementById("similar-refresh");
const incidentRefresh = document.getElementById("incident-refresh");
let lastQuery = "";
let lastService = "";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function apiFetch(url, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const payload = await response.json().catch(() => ({ detail: "Invalid server response." }));
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }
    return payload;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Request timed out.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function renderSimilarIncidents(items) {
  if (!items.length) {
    similarList.innerHTML = '<div class="empty-state">No similar incidents found.</div>';
    return;
  }

  similarList.innerHTML = items
    .map(
      (item) => `
        <article class="list-item">
          <h3>${escapeHtml(item.incident_id)} · ${escapeHtml(item.service_name)}</h3>
          <p>Similarity: ${escapeHtml(item.similarity)}</p>
          <p>${escapeHtml(item.resolution)}</p>
        </article>
      `,
    )
    .join("");
}

function renderIncidents(items) {
  if (!items.length) {
    incidentList.innerHTML = '<div class="empty-state">No incidents stored yet.</div>';
    return;
  }

  incidentList.innerHTML = items
    .slice(0, 20)
    .map(
      (item) => `
        <article class="list-item">
          <h3>${escapeHtml(item.incident_id)} · ${escapeHtml(item.service_name)}</h3>
          <p>${escapeHtml(item.error_message)}</p>
        </article>
      `,
    )
    .join("");
}

async function loadHealth() {
  try {
    const payload = await apiFetch("/health");
    healthPill.textContent = `${payload.status} · ${payload.incidents_count} incidents · ${payload.embedding_backend}`;
  } catch (error) {
    healthPill.textContent = "health check failed";
  }
}

async function loadIncidents() {
  incidentList.innerHTML = '<div class="empty-state">Loading incidents…</div>';
  try {
    const payload = await apiFetch("/incidents");
    renderIncidents(payload);
  } catch (error) {
    incidentList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

async function loadSimilar(query = lastQuery, serviceName = lastService) {
  if (!query) {
    similarList.innerHTML = '<div class="empty-state">Run RCA to load similar incidents.</div>';
    return;
  }

  similarList.innerHTML = '<div class="empty-state">Loading similar incidents…</div>';
  const params = new URLSearchParams({ message: query, top_k: "5" });
  if (serviceName) {
    params.set("service_name", serviceName);
  }

  try {
    const payload = await apiFetch(`/similar?${params.toString()}`);
    renderSimilarIncidents(payload);
  } catch (error) {
    similarList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

analyzeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  analyzeButton.disabled = true;
  analyzeError.textContent = "";
  analyzeButton.textContent = "Analyzing…";

  try {
    lastQuery = document.getElementById("log-input").value.trim();
    lastService = document.getElementById("service-input").value.trim();
    const payload = await apiFetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        log: lastQuery,
        service_name: lastService || null,
        top_k: 5,
      }),
    });

    resultEmpty.classList.add("hidden");
    resultContent.classList.remove("hidden");
    rootCause.textContent = payload.root_cause;
    severityPill.textContent = `Severity: ${payload.severity}`;
    confidencePill.textContent = `Confidence: ${payload.confidence_score}`;
    suggestedFix.textContent = payload.suggested_fix;
    renderSimilarIncidents(payload.similar_incidents || []);
  } catch (error) {
    analyzeError.textContent = error.message || "Load failed.";
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Run RCA";
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("file-input").files[0];
  if (!file) {
    uploadMessage.textContent = "Choose a file first.";
    uploadMessage.classList.add("error");
    return;
  }

  uploadButton.disabled = true;
  uploadButton.textContent = "Uploading…";
  uploadMessage.textContent = "";
  uploadMessage.classList.remove("error");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const payload = await apiFetch("/ingest", { method: "POST", body: formData });
    uploadMessage.textContent = `Ingested ${payload.logs_ingested} logs. Total incidents: ${payload.total_incidents}.`;
    await Promise.all([loadHealth(), loadIncidents()]);
  } catch (error) {
    uploadMessage.textContent = error.message || "Upload failed.";
    uploadMessage.classList.add("error");
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = "Upload";
  }
});

similarRefresh.addEventListener("click", () => {
  loadSimilar();
});

incidentRefresh.addEventListener("click", () => {
  loadIncidents();
});

window.addEventListener("load", async () => {
  await Promise.all([loadHealth(), loadIncidents()]);
});
