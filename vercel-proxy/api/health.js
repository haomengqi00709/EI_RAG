export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  if (req.method !== "GET") return res.status(405).end();

  const ENDPOINT = process.env.RUNPOD_ENDPOINT_ID;
  const API_KEY  = process.env.RUNPOD_API_KEY;

  try {
    const r = await fetch(`https://api.runpod.ai/v2/${ENDPOINT}/health`, {
      headers: { "Authorization": `Bearer ${API_KEY}` },
    });
    const data = await r.json();
    const w = data.workers || {};
    return res.status(200).json({
      idle:         w.idle         || 0,
      initializing: w.initializing || 0,
      ready:        w.ready        || 0,
      running:      w.running      || 0,
      throttled:    w.throttled    || 0,
    });
  } catch (e) {
    return res.status(200).json({ error: e.message });
  }
}
