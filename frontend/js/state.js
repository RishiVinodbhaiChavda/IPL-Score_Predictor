const STATE_KEY = "ipl_predictor_state";

function saveState(data) {
  localStorage.setItem(STATE_KEY, JSON.stringify(data));
}

function loadState() {
  try {
    return JSON.parse(localStorage.getItem(STATE_KEY)) || {};
  } catch {
    return {};
  }
}

function clearState() {
  localStorage.removeItem(STATE_KEY);
}
