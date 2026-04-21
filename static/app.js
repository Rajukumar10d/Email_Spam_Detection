const messageInput = document.getElementById("messageInput");
const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");
const sampleButton = document.getElementById("sampleButton");
const statusBadge = document.getElementById("statusBadge");
const predictionValue = document.getElementById("predictionValue");
const riskValue = document.getElementById("riskValue");
const spamValue = document.getElementById("spamValue");
const hamValue = document.getElementById("hamValue");
const featuresValue = document.getElementById("featuresValue");
const meterValue = document.getElementById("meterValue");
const progressBar = document.getElementById("progressBar");
const indicatorList = document.getElementById("indicatorList");
const cleanedText = document.getElementById("cleanedText");

const sampleMessage =
  "Congratulations! You have won a free vacation voucher. Click the link now to claim your reward.";

function setWaitingState() {
  statusBadge.textContent = "Waiting";
  statusBadge.className = "status-badge neutral";
  predictionValue.textContent = "-";
  riskValue.textContent = "Risk level: -";
  spamValue.textContent = "0.0000";
  hamValue.textContent = "0.0000";
  featuresValue.textContent = "0";
  meterValue.textContent = "0%";
  progressBar.style.width = "0%";
  indicatorList.innerHTML = '<li class="empty-state">Run an analysis to see indicator words.</li>';
  cleanedText.textContent = "-";
}

function renderIndicators(indicators) {
  if (!indicators.length) {
    indicatorList.innerHTML =
      '<li class="empty-state">No strong spam indicators matched the learned vocabulary.</li>';
    return;
  }

  indicatorList.innerHTML = indicators
    .map(
      (item) => `
        <li class="indicator-item">
          <div>
            <div class="indicator-token">${item.token}</div>
          </div>
          <div class="indicator-meta">
            count=${item.count}<br />
            weight=${item.spam_weight.toFixed(4)}
          </div>
        </li>
      `,
    )
    .join("");
}

async function analyzeMessage() {
  const message = messageInput.value.trim();
  if (!message) {
    statusBadge.textContent = "Input needed";
    statusBadge.className = "status-badge neutral";
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    const spamPercent = Math.round(data.spam_probability * 100);
    const isSpam = data.prediction === "spam";

    statusBadge.textContent = isSpam ? "Spam detected" : "Looks safe";
    statusBadge.className = `status-badge ${isSpam ? "spam" : "safe"}`;
    predictionValue.textContent = isSpam ? "SPAM" : "HAM";
    riskValue.textContent = `Risk level: ${data.risk_level}`;
    spamValue.textContent = data.spam_probability.toFixed(4);
    hamValue.textContent = data.ham_probability.toFixed(4);
    featuresValue.textContent = data.matched_features;
    meterValue.textContent = `${spamPercent}%`;
    progressBar.style.width = `${spamPercent}%`;
    renderIndicators(data.top_spam_indicators);
    cleanedText.textContent = data.cleaned_text || "-";
  } catch (error) {
    statusBadge.textContent = "Error";
    statusBadge.className = "status-badge spam";
    indicatorList.innerHTML = `<li class="empty-state">${error.message}</li>`;
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Detect Spam";
  }
}

analyzeButton.addEventListener("click", analyzeMessage);

clearButton.addEventListener("click", () => {
  messageInput.value = "";
  setWaitingState();
});

sampleButton.addEventListener("click", () => {
  messageInput.value = sampleMessage;
  messageInput.focus();
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    messageInput.value = chip.dataset.sample || "";
    messageInput.focus();
  });
});

messageInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    analyzeMessage();
  }
});

setWaitingState();
