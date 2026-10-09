"""Response Formatter and Output Parser Implementation."""

import json
import re
from typing import Any, Dict
from app.models.base import OutputParserABC


class ResponseFormatter(OutputParserABC):
    """
    Cleans up raw LLM and tool artifacts, parses markdown / JSON text,
    and structures user-facing responses.
    """

    def parse(self, raw_output: str) -> Dict[str, Any]:
        """
        Extracts structured data from raw textual output.
        Handles JSON codeblocks, markdown fences, or raw text.
        """
        if not raw_output:
            return {"text": "", "structured": {}}

        # Check for JSON markdown fences: ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(1))
                return {"text": raw_output, "structured": parsed_json}
            except json.JSONDecodeError:
                pass

        # Try parsing full string as JSON directly
        try:
            parsed_json = json.loads(raw_output.strip())
            return {"text": raw_output, "structured": parsed_json}
        except (json.JSONDecodeError, TypeError):
            pass

        return {"text": raw_output.strip(), "structured": {}}

    def format_final_output(self, raw_text: str, executed_tools_count: int = 0) -> str:
        """
        Strips internal control tags or thinking tags (e.g. <think>...</think> from reasoning models)
        to present a clean, definitive answer to the end user.
        """
        if not raw_text:
            return "Task completed without output."

        # Remove thinking blocks from DeepSeek-R1 or similar reasoning models
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        # Remove any lingering raw function tags
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL).strip()

        return cleaned


response_formatter = ResponseFormatter()
