export default async function handler(req, res) {
  // CORS headers so any browser can call this
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { question, fiscal_year, power_search, image } = req.body;
  if (!question) return res.status(400).json({ error: "No question provided" });

  const response = await fetch(
    `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/runsync`,
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.RUNPOD_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input: { question, fiscal_year, power_search, image },
      }),
    }
  );

  const data = await response.json();
  // RunPod wraps the result in { output: {...} } — unwrap it
  const result = data.output ?? data;

  // RunPod returns a status object (no output) when the worker is cold and
  // the job doesn't complete within runsync's wait window.
  if (!result.answer && result.status) {
    if (result.status === "IN_QUEUE" || result.status === "IN_PROGRESS") {
      return res.status(503).json({
        error: "The server is warming up after being idle. Please try again in about 60 seconds.",
      });
    }
    if (result.status === "FAILED") {
      return res.status(500).json({ error: result.error || "Request failed on the server." });
    }
  }

  return res.status(response.ok ? 200 : 500).json(result);
}
