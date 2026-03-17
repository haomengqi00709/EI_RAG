export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  if (req.method === "OPTIONS") return res.status(200).end();

  // Fire async job — returns immediately with job ID, worker starts loading in background
  const r = await fetch(
    `https://api.runpod.ai/v2/${process.env.RUNPOD_ENDPOINT_ID}/run`,
    {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.RUNPOD_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input: { ping: true } }),
    }
  );
  const data = await r.json();
  res.status(200).json({ jobId: data.id });
}
