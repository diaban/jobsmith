"""Real LLM clients implementing the framework protocols.

`AnthropicLLMClient` satisfies both `core.deps.LLMClient` (chat) and the
banking example's `VisionClient` (vision), backed by the official `anthropic`
SDK. Install with:  pip install -e ".[anthropic]"

Credentials resolve from the environment (ANTHROPIC_API_KEY, or an
`ant auth login` profile) — construct with no arguments for local dev.

Protocol impedance notes (LLMClient was shaped after an OpenAI-style API):
- `temperature` is accepted and IGNORED: sampling parameters are removed on
  Claude Opus 5 and sending them returns a 400.
- `response_format={"type": "json_object"}` has no direct equivalent; the
  prompts in this project already demand raw JSON, and the adapter
  defensively strips markdown fences from the reply.
- Safety-classifier refusals (`stop_reason: "refusal"`) raise RuntimeError so
  they flow through the framework's NodeError/escalation path. Server-side
  fallbacks are enabled by default (`fallbacks: "default"`), so a refusal is
  only surfaced after the fallback chain also declined.
"""
from __future__ import annotations

import base64
from typing import Any

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000


class AnthropicLLMClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        enable_fallbacks: bool = True,
        client: Any = None,        # injectable for tests
    ):
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
        self._client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.enable_fallbacks = enable_fallbacks

    # -------------------- helpers --------------------

    @staticmethod
    def _split_system(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """The Messages API takes the system prompt as a top-level param."""
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        if not rest:
            # Anthropic requires at least one user message
            rest = [{"role": "user", "content": "Proceed."}]
        return ("\n\n".join(system_parts) or None), rest

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Best-effort extraction of raw JSON from a fenced reply."""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3]
        return stripped.strip()

    @staticmethod
    def _text_of(response: Any) -> str:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RuntimeError(f"model refused the request (category={category})")
        return "".join(b.text for b in response.content if b.type == "text")

    async def _create(self, **kwargs: Any) -> Any:
        if self.enable_fallbacks:
            return await self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
                **kwargs,
            )
        return await self._client.messages.create(**kwargs)

    # -------------------- LLMClient protocol --------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,  # ignored — removed on Claude Opus 5
        max_tokens: int | None = None,
    ) -> str:
        system, rest = self._split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_output_tokens,
            "messages": rest,
        }
        if system:
            kwargs["system"] = system
        response = await self._create(**kwargs)
        text = self._text_of(response)
        if response_format and response_format.get("type") == "json_object":
            return self._strip_fences(text)
        return text

    # -------------------- VisionClient protocol --------------------

    async def vision(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        mime_type: str = "image/png",
    ) -> str:
        response = await self._create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return self._text_of(response)
