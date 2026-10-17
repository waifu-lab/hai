from typing import Any, AsyncIterator, Dict, Optional, Union

from httpx import AsyncClient

from .providers import ProviderInfo, ProviderTTS


class GPTSoVITS(ProviderInfo):
    """GPT-SoVITS server info and control."""

    def __init__(self, name: str = "GPTSoVITS", base_url: str = "http://127.0.0.1:9880"):
        """
        Args:
            name: Provider name.
            base_url: GPT-SoVITS WebAPI base URL (e.g. ``http://127.0.0.1:9880``).
        """
        super().__init__(name=name, api_key="", base_url=base_url)

    async def models(self) -> list:
        return []

    async def set_gpt_weights(self, weights_path: str) -> None:
        """Switch the active GPT model on the running server.

        Args:
            weights_path: Path to the ``.ckpt`` checkpoint file on the server.
        """
        async with AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/set_gpt_weights",
                params={"weights_path": weights_path},
            )
            response.raise_for_status()

    async def set_sovits_weights(self, weights_path: str) -> None:
        """Switch the active SoVITS model on the running server.

        Args:
            weights_path: Path to the ``.pth`` model file on the server.
        """
        async with AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/set_sovits_weights",
                params={"weights_path": weights_path},
            )
            response.raise_for_status()

    async def control(self, command: str) -> None:
        """Send a control command to the server.

        Args:
            command: ``"restart"`` to reload the server, ``"exit"`` to shut it down.
        """
        async with AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/control",
                json={"command": command},
            )
            response.raise_for_status()


class GPTSoVITSTTS(ProviderTTS):
    """Text-to-speech via a local GPT-SoVITS WebAPI server.

    Requests go to ``{base_url}/tts`` and return WAV audio.
    """

    def __init__(
        self,
        name: str = "GPTSoVITSTTS",
        base_url: str = "http://127.0.0.1:9880",
        ref_audio_path: str = "",
        prompt_text: str = "",
        prompt_lang: str = "zh",
        text_lang: str = "zh",
        text_split_method: str = "cut5",
        batch_size: int = 1,
        speed_factor: float = 1.0,
        streaming_mode: Union[bool, int] = False,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        timeout: float = 60.0,
    ):
        """
        Args:
            name: Provider name.
            base_url: GPT-SoVITS WebAPI base URL (e.g. ``http://127.0.0.1:9880``).
            ref_audio_path: Path to the reference audio file used for voice cloning.
            prompt_text: Transcript of the reference audio (improves quality).
            prompt_lang: Language of ``prompt_text``. E.g. ``"zh"``, ``"en"``, ``"ja"``.
            text_lang: Language of the text to synthesize.
            text_split_method: How to split long text before inference. Options:

                * ``"cut1"`` — group every 4 sentences
                * ``"cut2"`` — group every 50 characters
                * ``"cut3"`` — split on Chinese period ``。``
                * ``"cut4"`` — split on English period ``.``
                * ``"cut5"`` — split on all punctuation marks *(default)*
            batch_size: Number of text segments processed per inference batch.
            speed_factor: Playback speed multiplier (1.0 = normal speed).
            streaming_mode: Controls audio streaming behaviour.
                ``False``/``0`` disables streaming; ``True``/``1`` streams with best
                quality (slowest); ``2`` medium quality; ``3`` lowest quality (fastest).
                Used only by :meth:`synthesize_stream`.
            top_k: Top-k sampling for the T2S model.
            top_p: Top-p (nucleus) sampling for the T2S model.
            temperature: Sampling temperature for the T2S model.
            timeout: HTTP request timeout in seconds.
        """
        super().__init__(name=name, base_url=base_url, timeout=timeout)
        self.ref_audio_path = ref_audio_path
        self.prompt_text = prompt_text
        self.prompt_lang = prompt_lang
        self.text_lang = text_lang
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.speed_factor = speed_factor
        self.streaming_mode = streaming_mode
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    @property
    def tts_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/tts"

    def _build_payload(self, text: str, streaming: Union[bool, int] = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_lang": self.prompt_lang,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "speed_factor": self.speed_factor,
            "streaming_mode": streaming,
        }
        if self.prompt_text:
            payload["prompt_text"] = self.prompt_text
        return payload

    async def synthesize(self, text: str) -> bytes:
        """Synthesize speech and return the complete WAV audio as bytes.

        Args:
            text: Text to synthesize.

        Returns:
            Raw WAV audio bytes.
        """
        payload = self._build_payload(text, streaming=False)
        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.tts_url, json=payload)
            response.raise_for_status()
            return response.content

    async def synthesize_stream(self, text: str, chunk_size: int = 4096) -> "AsyncIterator[bytes]":
        """Synthesize speech and yield WAV audio in chunks.

        Uses :attr:`streaming_mode` if set (defaults to ``1`` — best quality).

        Args:
            text: Text to synthesize.
            chunk_size: Number of bytes per yielded chunk.

        Yields:
            Raw WAV audio byte chunks.
        """
        streaming = self.streaming_mode if self.streaming_mode else 1
        payload = self._build_payload(text, streaming=streaming)
        async with AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", self.tts_url, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size):
                    if chunk:
                        yield chunk
