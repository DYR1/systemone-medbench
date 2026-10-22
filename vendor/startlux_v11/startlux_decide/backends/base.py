"""The interface every inference engine implements: two primitives and the tokenizer.

``readout(prompt, letter_ids)``
    One forward pass over the prompt; the log-probabilities (or logits: only their differences are used) of the given
    letter tokens as the next token, in the order of `letter_ids`.
``generate(prompt, max_tokens, sampling, stop, seed)``
    Sampled continuation of the prompt, ended by a stop string (not included in the text), the end-of-sequence token
    or `max_tokens`. Only AutoThink models generate (their thoughts).

Engines that read images set ``reads_images``: their primitives then also take ``media`` (images.Media, the images
of the request), and the prompt holds the image placeholder of the chat template at each image's place. Engines that
keep a prefix cache set ``caches_prefixes`` and implement ``prefill(prompt, media)``, which reads a prompt into the
cache without answering anything (the questions of a request that share a long state then read it once).

Engines that can stream a thought safely set ``streams`` and implement a third primitive:

``generate_stream(prompt, max_tokens, sampling, stop, seed)``
    The same continuation as one request, delivered as it is generated: an async iterator of StreamEvent. The last
    event has a finish_reason; a caller that stops iterating early must close the iterator (``aclose()``), which ends
    the generation in the engine.

Prompts are text rendered with the model folder's chat template; ``tokenizer`` is the folder's Hugging Face tokenizer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence


@dataclass(frozen=True)
class Sampling:
    """Sampling settings of a thought."""
    temperature: float
    top_p: float
    top_k: int


@dataclass(frozen=True)
class Generation:
    """What `generate` returns. finish_reason is "stop" (stop string or end-of-sequence token) or "length"."""
    text: str
    tokens: int
    finish_reason: str


@dataclass(frozen=True)
class StreamEvent:
    """One event of `generate_stream`: the text added since the previous event, the tokens generated so far, and on
    the last event the finish_reason ("stop" or "length"; None before)."""
    text: str
    tokens: int
    finish_reason: str | None = None


class BackendError(RuntimeError):
    """The engine could not be reached or failed (the server answers 502)."""


class BackendRequestError(ValueError):
    """The engine refused this request, for example a prompt longer than its context (the server answers 422)."""


class Backend(ABC):
    """Base class of the engines. Subclasses set `name` and `tokenizer` and implement the two primitives."""

    name = "base"
    tokenizer: Any = None
    context_length: int | None = None       # tokens of prompt + generation the engine accepts, when it says
    streams: bool = False                   # implements generate_stream and may use it for thoughts
    reads_images: bool = False              # the primitives take media= (images in the prompt)
    caches_prefixes: bool = False           # prompts that share a beginning reuse it; prefill() fills the cache

    async def start(self) -> None:
        """Connect to or load the engine (called once before the first request)."""

    async def close(self) -> None:
        """Release connections, threads and processes."""

    def lane(self) -> "Backend":
        """The backend to use for every call of one question. Engines with several servers pin a question to one
        server (round robin over questions); the default is the backend itself."""
        return self

    @abstractmethod
    async def readout(self, prompt: str, letter_ids: Sequence[int]) -> list[float]:
        """Next-token log-probabilities of the letters after the prompt, in the order of letter_ids."""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                       seed: int) -> Generation:
        """Sampled continuation of the prompt (see the module docstring)."""

    def generate_stream(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                        seed: int) -> AsyncIterator[StreamEvent]:
        """The continuation of `generate` as one streamed request (engines with `streams` only)."""
        raise NotImplementedError(f"the {self.name} engine does not stream thoughts")

    async def prefill(self, prompt: str, media: Any = None) -> None:
        """Read a prompt into the engine's prefix cache (engines with caches_prefixes); otherwise nothing."""

    def describe(self) -> dict:
        """What /health reports about the engine."""
        return {"name": self.name}
