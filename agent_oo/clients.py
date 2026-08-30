"""Real LLM clients implementing the framework protocols.

Both classes satisfy `core.deps.LLMClient` (chat) and the banking example's
`VisionClient` (vision):

- `AnthropicLLMClient` — official `anthropic` SDK.  pip install -e ".[anthropic]"
  Credentials from ANTHROPIC_API_KEY or an `ant auth login` profile.
- `OpenAILLMClient` — official `openai` SDK.        pip install -e ".[openai]"
  Credentials from OPENAI_API_KEY; OPENAI_BASE_URL also works, so any
  OpenAI-compatible endpoint (Azure, Ollama, vLLM, a corporate gateway) can
  be targeted without code changes.

Construct either with no arguments for local dev.

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


DEFAULT_OPENAI_MODEL = "gpt-5.1"

# Reasoning-model families: temperature is rejected (only the default is
# allowed) and the output cap is `max_completion_tokens`, not `max_tokens`.
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class OpenAILLMClient:
    """LLMClient/VisionClient backed by an OpenAI-compatible chat API.

    The framework's LLMClient protocol is OpenAI-shaped, so this is mostly a
    pass-through. Two model-dependent quirks are handled automatically for
    reasoning models (gpt-5*/o*): `temperature` is dropped (the API rejects
    non-default values) and the cap is sent as `max_completion_tokens`.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any = None,        # injectable for tests
    ):
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
        self._client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    # -------------------- helpers --------------------

    @property
    def _is_reasoning_model(self) -> bool:
        return self.model.startswith(_OPENAI_REASONING_PREFIXES)

    def _base_kwargs(self, max_tokens: int | None) -> dict[str, Any]:
        cap = max_tokens or self.max_output_tokens
        if self._is_reasoning_model:
            return {"model": self.model, "max_completion_tokens": cap}
        return {"model": self.model, "max_tokens": cap}

    @staticmethod
    def _text_of(response: Any) -> str:
        message = response.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"model refused the request: {refusal}")
        return message.content or ""

    # -------------------- LLMClient protocol --------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        kwargs = self._base_kwargs(max_tokens)
        kwargs["messages"] = messages
        if response_format is not None:
            kwargs["response_format"] = response_format  # json_object maps 1:1
        if not self._is_reasoning_model:
            kwargs["temperature"] = temperature
        response = await self._client.chat.completions.create(**kwargs)
        return self._text_of(response)

    # -------------------- VisionClient protocol --------------------

    async def vision(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        mime_type: str = "image/png",
    ) -> str:
        data_uri = (
            f"data:{mime_type};base64,"
            + base64.standard_b64encode(image_bytes).decode("utf-8")
        )
        kwargs = self._base_kwargs(None)
        kwargs["messages"] = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }]
        response = await self._client.chat.completions.create(**kwargs)
        return self._text_of(response)
