const modelScores = [
  { name: "Logistic Regression", shortName: "Logistic Reg.", accuracy: 0.8077, precision: 0.6868, recall: 0.2396, auc: 0.7076 },
  { name: "Random Forest", shortName: "Random Forest", accuracy: 0.8148, precision: 0.6530, recall: 0.3474, auc: 0.7563 },
  { name: "XGBoost", shortName: "XGBoost", accuracy: 0.8175, precision: 0.6620, recall: 0.3572, auc: 0.7782 },
];

const metricInfo = {
  auc: { label: "ROC-AUC", description: "Ability to rank defaulters above non-defaulters across thresholds." },
  accuracy: { label: "Accuracy", description: "Share of all test-set predictions that are correct." },
  precision: { label: "Precision", description: "Share of clients flagged as defaulters who actually default." },
  recall: { label: "Recall", description: "Share of actual defaulters the model identifies." },
};

const barContainer = document.querySelector("#metric-bars");
const metricDescription = document.querySelector("#metric-description");
const metricButtons = document.querySelectorAll("[data-metric]");
const formatScore = (value) => value.toFixed(4);
const STORAGE_KEY = "credit-card-default-dashboard-metrics-v1";
let activeMetric = "auc";
let savedState = {};

try {
  savedState = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
} catch {
  savedState = {};
}

modelScores.forEach((model) => {
  const saved = savedState.scores?.[model.name];
  if (saved) {
    Object.keys(metricInfo).forEach((metric) => {
      const value = Number(saved[metric]);
      if (Number.isFinite(value) && value >= 0 && value <= 1) model[metric] = value;
    });
  }
});

function saveState() {
  const scores = Object.fromEntries(modelScores.map((model) => [model.name, {
    auc: model.auc, accuracy: model.accuracy, precision: model.precision, recall: model.recall,
  }]));
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ scores, order: savedState.order || {} }));
  } catch {
    // Keep the chart usable if browser storage is unavailable.
  }
}

function getOrder(metric) {
  const requested = savedState.order?.[metric];
  if (!Array.isArray(requested)) return [...modelScores].sort((a, b) => b[metric] - a[metric]).map((model) => model.name);
  const valid = requested.filter((name) => modelScores.some((model) => model.name === name));
  return [...valid, ...modelScores.map((model) => model.name).filter((name) => !valid.includes(name))];
}

function updateOrder(names) {
  savedState.order ||= {};
  savedState.order[activeMetric] = names;
  saveState();
}

function renderMetric(metric) {
  if (!barContainer || !metricInfo[metric]) return;
  activeMetric = metric;
  const ranked = getOrder(metric).map((name) => modelScores.find((model) => model.name === name));
  metricDescription.textContent = metricInfo[metric].description;
  barContainer.replaceChildren();

  ranked.forEach((model, index) => {
    const row = document.createElement("div");
    row.className = `metric-row${index === 0 ? " is-top" : ""}`;
    row.draggable = true;
    row.dataset.model = model.name;
    row.title = "Drag to reorder this model";
    row.setAttribute("aria-label", `${model.name}: ${metricInfo[metric].label} ${formatScore(model[metric])}`);

    row.addEventListener("dragstart", (event) => {
      row.classList.add("is-dragging");
      event.dataTransfer.setData("text/plain", model.name);
      event.dataTransfer.effectAllowed = "move";
    });
    row.addEventListener("dragend", () => row.classList.remove("is-dragging"));
    row.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    });
    row.addEventListener("drop", (event) => {
      event.preventDefault();
      const draggedName = event.dataTransfer.getData("text/plain");
      if (!draggedName || draggedName === model.name) return;
      const next = [...barContainer.querySelectorAll(".metric-row")].map((item) => item.dataset.model);
      const from = next.indexOf(draggedName);
      const to = next.indexOf(model.name);
      if (from < 0 || to < 0) return;
      next.splice(to, 0, next.splice(from, 1)[0]);
      updateOrder(next);
      renderMetric(activeMetric);
    });

    const moveControls = document.createElement("span");
    moveControls.className = "metric-move-controls";
    [
      { label: "Move up", symbol: "↑", delta: -1 },
      { label: "Move down", symbol: "↓", delta: 1 },
    ].forEach(({ label, symbol, delta }) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "metric-move";
      button.textContent = symbol;
      button.setAttribute("aria-label", `${label} ${model.name}`);
      button.disabled = index + delta < 0 || index + delta >= ranked.length;
      button.addEventListener("click", () => {
        const next = [...barContainer.querySelectorAll(".metric-row")].map((item) => item.dataset.model);
        const target = index + delta;
        [next[index], next[target]] = [next[target], next[index]];
        updateOrder(next);
        renderMetric(activeMetric);
      });
      moveControls.append(button);
    });

    const name = document.createElement("span");
    name.className = "metric-name";
    name.textContent = model.shortName;

    const track = document.createElement("span");
    track.className = "metric-track";
    track.setAttribute("role", "img");
    track.setAttribute("aria-label", `${(model[metric] * 100).toFixed(2)} percent`);

    const fill = document.createElement("i");
    fill.className = "metric-fill";
    fill.style.width = `${model[metric] * 100}%`;
    track.append(fill);

    const value = document.createElement("span");
    value.className = "metric-value-wrap";
    const input = document.createElement("input");
    input.className = "metric-value";
    input.type = "number";
    input.min = "0";
    input.max = "1";
    input.step = "0.0001";
    input.value = formatScore(model[metric]);
    input.setAttribute("aria-label", `${model.name} ${metricInfo[metric].label} score, from 0 to 1`);
    input.addEventListener("change", () => {
      const nextValue = Number(input.value);
      if (!Number.isFinite(nextValue) || nextValue < 0 || nextValue > 1) {
        input.value = formatScore(model[metric]);
        return;
      }
      model[metric] = nextValue;
      saveState();
      renderMetric(activeMetric);
    });
    value.append(input);

    row.append(moveControls, name, track, value);
    barContainer.append(row);
  });
}

metricButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const metric = button.dataset.metric;
    metricButtons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    renderMetric(metric);
  });
});

renderMetric("auc");

