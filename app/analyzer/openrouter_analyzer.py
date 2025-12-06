import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple

import requests

from app.models import TimeRange, hhmmss_to_seconds
from .base import VisualAnalyzerBase


logger = logging.getLogger(__name__)


class OpenRouterVideoAnalyzer(VisualAnalyzerBase):
    """
    Vision language model caller that asks for answer-aligned time ranges.
    Raises on failure; caller can decide how to handle errors.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None, max_frames: int = 500):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterVideoAnalyzer.")
        self.model = model or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-pro")
        self.max_frames = max_frames
        logger.info("OpenRouterVideoAnalyzer configured with model=%s max_frames=%d", self.model, self.max_frames)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _sample_frames(self, frame_paths: List[str]) -> List[str]:
        if len(frame_paths) <= self.max_frames:
            return frame_paths
        step = len(frame_paths) / self.max_frames
        return [frame_paths[int(i * step)] for i in range(self.max_frames)]

    def _build_messages(self, frame_paths: List[str], user_prompt: str) -> List[dict]:
        system_prompt = (
            "You are a video question-answering expert. The input video is represented by a series of frames provided in temporal order. "
            "Find the exact time ranges in the video where the user's request is satisfied. Think carefully before answering. Only return the time ranges when you are confident they are correct. "
            "Only return JSON with start and end times. Make sure end is bigger than start."
        )
        user_text = (
            f"User prompt: {user_prompt}\n"
            "First look through all the provided frames in temporal order and then return the most matching 0 - 5 concise ranges "
            "in the format HH:MM:SS.mmm where the answer appears."
        )

        content: List[dict] = [{"type": "text", "text": user_text}]
        for frame_path in frame_paths:
            content.append({"type": "text", "text": f"Frame from {Path(frame_path).name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(frame_path)},
                }
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _response_schema(self) -> dict:
        return {
            "name": "answer_ranges",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ranges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string", "description": "HH:MM:SS.mmm"},
                                "end": {"type": "string", "description": "HH:MM:SS.mmm"},
                            },
                            "required": ["start", "end"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                        "maxItems": 4,
                    },
                    "notes": {"type": "string"},
                },
                "required": ["ranges"],
                "additionalProperties": False,
            },
        }

    def analyze(
        self,
        preprocess_result,
        user_prompt: str,
        response_dump_path: Path | None = None,
        request_dump_path: Path | None = None,
    ) -> Tuple[List[TimeRange], str]:
        frame_paths = list(preprocess_result.frames.values())
        if not frame_paths:
            raise RuntimeError("No frames available for VLM analysis")

        sampled_frames = self._sample_frames(frame_paths)
        logger.info("Analyzing %d frames (sampled from %d)", len(sampled_frames), len(frame_paths))
        messages = self._build_messages(sampled_frames, user_prompt)

        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": self._response_schema()},
            "reasoning": {
                "effort": "high"
            }
        }

        if request_dump_path:
            try:
                request_dump_path.parent.mkdir(parents=True, exist_ok=True)
                with open(request_dump_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                logger.info("Saved analyzer request to %s", request_dump_path)
            except Exception as dump_err:
                logger.warning("Failed to write analyzer request: %s", dump_err)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = None
        try:
            response = requests.post(self.API_URL, headers=headers, json=payload, timeout=600)
            logger.info("OpenRouter POST status: %s", response.status_code)
            logger.debug("OpenRouter response text (first 500): %s", response.text[:500])
            response.raise_for_status()
        except requests.HTTPError as http_err:
            body = response.text[:500] if response is not None else ""
            logger.error("OpenRouter HTTP error %s, body: %s", getattr(response, "status_code", "?"), body)
            raise
        except requests.RequestException as req_err:
            logger.error("OpenRouter request failed: %s", req_err, exc_info=True)
            raise

        try:
            data = response.json()
        except json.JSONDecodeError as json_err:
            logger.error("Failed to decode OpenRouter JSON: %s", json_err)
            logger.error("Response text (first 500): %s", response.text[:500] if response else "")
            raise

        content = data["choices"][0]["message"]["content"]
        logger.info("Raw content length: %d", len(content))

        # Optionally dump raw JSON response for debugging/tracing.
        if response_dump_path:
            try:
                response_dump_path.parent.mkdir(parents=True, exist_ok=True)
                with open(response_dump_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                logger.info("Saved analyzer raw response to %s", response_dump_path)
            except Exception as dump_err:
                logger.warning("Failed to write analyzer raw response: %s", dump_err)

        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"```$", "", content).strip()

        parsed = json.loads(content)
        ranges_raw = parsed.get("ranges", [])
        ranges = [TimeRange(start=hhmmss_to_seconds(r["start"]), end=hhmmss_to_seconds(r["end"])) for r in ranges_raw]

        note = parsed.get("notes", "Used OpenRouter vision model.")
        logger.info("Parsed %d ranges", len(ranges))
        return ranges, note
