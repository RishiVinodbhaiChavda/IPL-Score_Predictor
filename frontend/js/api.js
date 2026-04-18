const API_BASE = "http://localhost:8000/api";

async function fetchTeams() {
  const r = await fetch(`${API_BASE}/teams`);
  return r.json();
}

async function fetchPlayers(teamId) {
  const r = await fetch(`${API_BASE}/players?team_id=${teamId}`);
  return r.json();
}

async function fetchVenues() {
  const r = await fetch(`${API_BASE}/venues`);
  return r.json();
}

async function predictScore(payload) {
  const r = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return r.json();
}

function showToast(msg, type = "") {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg;
  t.className = `toast ${type} show`;
  setTimeout(() => t.classList.remove("show"), 3000);
}
