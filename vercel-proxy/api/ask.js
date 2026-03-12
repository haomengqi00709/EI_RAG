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
  // RunPod wraps the result in { output: {...} } — unwrap it.
  // Generator handlers return output as an array of all yielded values;
  // take the last element which is the final result dict.
  let result = data.output || data;
  if (Array.isArray(result)) {
    result = result[result.length - 1] || {};
  }
  // Strip the {type:"result"} wrapper added for streaming compatibility
  if (result.type === "result") {
    const { type: _t, ...rest } = result;
    result = rest;
  }
  return res.status(response.ok ? 200 : 500).json(result);
}
