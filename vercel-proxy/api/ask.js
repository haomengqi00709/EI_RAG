export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { question, fiscal_year, power_search, image } = req.body;
  if (!question) return res.status(400).json({ error: "No question provided" });

  const ENDPOINT = process.env.RUNPOD_ENDPOINT_ID;
  const API_KEY  = process.env.RUNPOD_API_KEY;

  // ── SSE setup ──────────────────────────────────────────────────────────────
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");

  function send(data) {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  }

  // ── Submit async job to RunPod ─────────────────────────────────────────────
  let jobId;
  try {
    const runResp = await fetch(`https://api.runpod.ai/v2/${ENDPOINT}/run`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input: { question, fiscal_year, power_search, image } }),
    });
    const runData = await runResp.json();
    jobId = runData.id;
    if (!jobId) {
      send({ type: "error", error: "Failed to submit job: " + JSON.stringify(runData) });
      return res.end();
    }
  } catch (e) {
    send({ type: "error", error: "Failed to connect to RunPod: " + e.message });
    return res.end();
  }

  send({ type: "progress", message: "Connecting to worker\u2026" });

  // ── Poll /stream/{jobId} and forward new chunks as SSE ────────────────────
  // RunPod returns ALL accumulated stream chunks on each poll, so we track seenCount.
  let seenCount = 0;
  const MAX_POLLS = 200; // ~3.3 min at 1 s intervals

  for (let i = 0; i < MAX_POLLS; i++) {
    await new Promise(r => setTimeout(r, 1000));

    let streamData;
    try {
      const streamResp = await fetch(
        `https://api.runpod.ai/v2/${ENDPOINT}/stream/${jobId}`,
        { headers: { "Authorization": `Bearer ${API_KEY}` } }
      );
      streamData = await streamResp.json();
    } catch (_) {
      continue; // transient error — keep polling
    }

    // Forward only new chunks
    const streamChunks = streamData.stream || [];
    for (let j = seenCount; j < streamChunks.length; j++) {
      if (streamChunks[j].output) send(streamChunks[j].output);
    }
    seenCount = streamChunks.length;

    if (streamData.status === "COMPLETED") {
      // If final result wasn't in the stream chunks, it lives in streamData.output
      const hasResult = streamChunks.some(c => c.output && c.output.type === "result");
      if (!hasResult && streamData.output) send(streamData.output);
      break;
    }

    if (streamData.status === "FAILED") {
      send({ type: "error", error: streamData.error || "Job failed on RunPod" });
      break;
    }
  }

  res.end();
}
