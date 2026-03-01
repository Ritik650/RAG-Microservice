"""Answer generation with Google Gemini, grounded in retrieved context.

Uses the google-genai SDK's async client. The client is created lazily so importing
this module (and running tests) does not require GEMINI_API_KEY or the SDK package.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

NO_CONTEXT = "I couldn't find anything relevant in the indexed documents to answer that."

SYSTEM = (
    "You answer questions strictly from the provided context passages. "
    "Cite the passages you use with bracketed numbers like [1], [2] that map to the "
    "numbered CONTEXT blocks. If the context does not contain the answer, say you don't know. "
    "Be concise and never invent sources."
)


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        header = f"[{i}] (source: {c['source']}#chunk{c['chunk_index']})"
        blocks.append(f"{header}\n{c['text']}")
    context = "\n\n".join(blocks)
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above and cite the passages you used with [n]."
    )


class GeminiGenerator:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            # Falls back to the GEMINI_API_KEY env var if not set in Settings.
            self._client = genai.Client(api_key=self.settings.gemini_api_key or None)
        return self._client

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=self.settings.llm_max_tokens,
        )

    async def generate(self, question: str, contexts: list[dict]) -> str:
        if not contexts:
            return NO_CONTEXT
        response = await self.client.aio.models.generate_content(
            model=self.settings.llm_model,
            contents=build_user_prompt(question, contexts),
            config=self._config(),
        )
        return (response.text or "").strip()

    async def stream(self, question: str, contexts: list[dict]) -> AsyncIterator[str]:
        if not contexts:
            yield NO_CONTEXT
            return
        stream = await self.client.aio.models.generate_content_stream(
            model=self.settings.llm_model,
            contents=build_user_prompt(question, contexts),
            config=self._config(),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
