export default async function handler(req, res) {
  // CORS headers so any browser can call this
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { question, fiscal_year, power_search } = req.body;
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
        input: { question, fiscal_year, power_search },
      }),
    }
  );

  const data = await response.json();
  return res.status(response.ok ? 200 : 500).json(data);
}
