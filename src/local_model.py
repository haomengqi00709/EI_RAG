"""
local_model.py — Drop-in wrapper for a local MLX generative model.

Provides the same .generate_content(prompt, generation_config) interface
as google.generativeai.GenerativeModel so it can be swapped in server.py
without touching any other code.

Usage (via ANSWER_MODEL env var in .env):
    ANSWER_MODEL=local   → uses Qwen/Qwen3-4B-MLX-4bit locally
    ANSWER_MODEL=gemini  → uses Gemini 2.0 Flash (default)
"""

import json
import re

from mlx_lm import generate as mlx_generate
from mlx_lm import load


class _Response:
    """Mirrors the response object returned by Gemini's generate_content."""
    def __init__(self, text: str):
        self.text = text


class LocalMLXModel:
    """
    Wraps a local MLX generative model with the same interface as
    google.generativeai.GenerativeModel.

    The model is loaded lazily on the first generate_content() call
    so startup time is unaffected.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-MLX-4bit",
        system_instruction: str | None = None,
        max_tokens: int = 512,
    ):
        self.model_id        = model_id
        self.system_instruction = system_instruction
        self.max_tokens      = max_tokens
        self._model          = None
        self._tokenizer      = None

    def _load(self):
        if self._model is None:
            print(f"[local_model] Loading {self.model_id}…", flush=True)
            self._model, self._tokenizer = load(self.model_id)
            print("[local_model] Model ready.", flush=True)

    def generate_content(self, prompt: str, generation_config: dict | None = None) -> _Response:
        self._load()

        want_json = (generation_config or {}).get("response_mime_type") == "application/json"

        # Build chat messages
        messages = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})

        user_content = prompt
        if want_json:
            user_content += "\n\nReturn ONLY valid JSON — no markdown fences, no explanation."
        messages.append({"role": "user", "content": user_content})

        # Apply Qwen3 chat template (disable thinking mode for speed)
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizer versions don't support enable_thinking
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        raw = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=text,
            max_tokens=self.max_tokens,
            verbose=False,
        )

        result = raw.strip()

        # For JSON mode: strip any markdown fences the model added anyway
        if want_json:
            result = _extract_json(result)

        return _Response(result)


def _extract_json(text: str) -> str:
    """
    Try to pull a clean JSON object/array out of model output that may
    include markdown fences or surrounding prose.
    """
    # Remove ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        return fenced.group(1).strip()

    # Find the first { or [ and take everything from there
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        idx = text.find(start_char)
        if idx != -1:
            # Find matching close
            candidate = text[idx:]
            # Quick validation: try to parse
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                # Try trimming to last closing brace
                last = candidate.rfind(end_char)
                if last != -1:
                    try:
                        json.loads(candidate[:last + 1])
                        return candidate[:last + 1]
                    except json.JSONDecodeError:
                        pass

    return text  # fallback: return as-is and let the caller handle parse errors
