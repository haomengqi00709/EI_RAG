const RUNPOD_BASE = `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}`;
const AUTH = `Bearer ${process.env.RUNPOD_API_KEY}`;

async function pollJob(jobId, timeoutMs = 100000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 4000));
    const r = await fetch(`${RUNPOD_BASE}/status/${jobId}`, {
      headers: { Authorization: AUTH },
    });
    const data = await r.json();
    if (data.status === "COMPLETED") return data.output ?? data;
    if (data.status === "FAILED")    return { error: data.error || "Request failed on the server." };
  }
  return { error: "Request timed out. The server is still warming up — please try again." };
}

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { question, fiscal_year, power_search, image } = req.body;
  if (!question) return res.status(400).json({ error: "No question provided" });

  const response = await fetch(`${RUNPOD_BASE}/runsync`, {
    method: "POST",
    headers: { Authorization: AUTH, "Content-Type": "application/json" },
    body: JSON.stringify({ input: { question, fiscal_year, power_search, image } }),
  });

  const data = await response.json();
  const result = data.output ?? data;

  // Job completed within runsync window
  if (result.answer || (!result.status)) {
    return res.status(response.ok ? 200 : 500).json(result);
  }

  // Worker cold — poll until done
  if (result.status === "IN_QUEUE" || result.status === "IN_PROGRESS") {
    const jobId = data.id;
    const polled = await pollJob(jobId);
    return res.status(polled.error ? 503 : 200).json(polled);
  }

  if (result.status === "FAILED") {
    return res.status(500).json({ error: result.error || "Request failed on the server." });
  }

  return res.status(response.ok ? 200 : 500).json(result);
}
