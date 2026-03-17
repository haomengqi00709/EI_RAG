import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json({ limit: "20mb" }));
app.use(express.static(path.join(__dirname, "public")));

const RUNPOD_BASE = `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}`;
const AUTH = `Bearer ${process.env.RUNPOD_API_KEY}`;

// ── Health ─────────────────────────────────────────────────────────────────
app.get("/api/health", async (req, res) => {
  try {
    const r = await fetch(`${RUNPOD_BASE}/health`, { headers: { Authorization: AUTH } });
    res.json(await r.json());
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Warmup ─────────────────────────────────────────────────────────────────
app.post("/api/warmup", async (req, res) => {
  try {
    const r = await fetch(`${RUNPOD_BASE}/run`, {
      method: "POST",
      headers: { Authorization: AUTH, "Content-Type": "application/json" },
      body: JSON.stringify({ input: { ping: true } }),
    });
    const data = await r.json();
    res.json({ jobId: data.id });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Ask ────────────────────────────────────────────────────────────────────
app.post("/api/ask", async (req, res) => {
  const { question, fiscal_year, power_search, image } = req.body;
  if (!question) return res.status(400).json({ error: "No question provided" });

  // Try runsync first (fast path for warm workers)
  const response = await fetch(`${RUNPOD_BASE}/runsync`, {
    method: "POST",
    headers: { Authorization: AUTH, "Content-Type": "application/json" },
    body: JSON.stringify({ input: { question, fiscal_year, power_search, image } }),
  });

  const data = await response.json();
  const result = data.output ?? data;

  if (result.answer || !result.status) return res.json(result);
  if (result.status === "FAILED") return res.status(500).json({ error: result.error || "Request failed." });

  // Cold start — poll until done (no timeout issue on Railway)
  if (result.status === "IN_QUEUE" || result.status === "IN_PROGRESS") {
    const jobId = data.id;
    const deadline = Date.now() + 110_000;
    while (Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 4000));
      const poll = await fetch(`${RUNPOD_BASE}/status/${jobId}`, { headers: { Authorization: AUTH } });
      const pd   = await poll.json();
      if (pd.status === "COMPLETED") return res.json(pd.output ?? pd);
      if (pd.status === "FAILED")    return res.status(500).json({ error: pd.error || "Request failed." });
    }
    return res.status(503).json({ error: "Request timed out. Please try again." });
  }

  return res.status(response.ok ? 200 : 500).json(result);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on port ${PORT}`));
