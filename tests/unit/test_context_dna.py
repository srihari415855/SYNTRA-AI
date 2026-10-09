"""Unit tests for Context DNA State Preservation and Token-Aware Compression."""

import unittest
from app.services.context_dna import (
    ContextDNASchema,
    HandoverPacket,
    context_dna_manager,
)


class TestContextDNA(unittest.TestCase):
    """Validates 8 canonical components and compression logic in Context DNA."""

    def test_extract_dna_components(self):
        """Extracts Goal, Constraints, State, Knowledge, Artifacts from messages."""
        messages = [
            {"role": "user", "content": "Write a Python script to sort records into JSON format"},
            {"role": "assistant", "content": "```python\ndef sort_records(recs):\n    return sorted(recs)\n```"},
            {"role": "user", "content": "Now add filtering for active records"}
        ]
        tools = [{"tool_name": "system_info", "output": {"cpu_percent": 12.5}}]

        dna = context_dna_manager.extract_dna_from_session(
            session_id="test_sess_1",
            messages=messages,
            tools_executed=tools,
            last_failure="Provider 503 Outage"
        )

        # Component A: Goal
        self.assertIn("filtering", dna.objective.lower())

        # Component B: Constraints
        self.assertIn("JSON structured format", dna.format_requirements)
        self.assertTrue(len(dna.explicit_requirements) > 0)

        # Component C: Current state
        self.assertEqual(dna.current_task, "Model Handover & Continuation")

        # Component D: Verified knowledge
        self.assertTrue(any("system_info" in f for f in dna.confirmed_findings))

        # Component E: Uncertainty and defects
        self.assertIn("Provider 503 Outage", dna.failed_validations)

        # Component G: Artifacts (real code preservation)
        self.assertTrue(len(dna.code_artifacts) > 0)
        self.assertTrue(any("def sort_records" in code for code in dna.code_artifacts.values()))

    def test_compression_and_retention_metrics(self):
        """Tests token compression metrics, constraint retention, and budget checks."""
        dna = ContextDNASchema(
            objective="Develop an image classification pipeline",
            intended_outcome="Working pipeline with tests",
            current_task="Model Handover",
            immediate_next_action="Implement validation step",
            continuation_prompt="Complete implementation",
            explicit_requirements=["Use PyTorch", "Include batch norm"],
            code_artifacts={"pipeline.py": "class Pipeline:\n    pass\n" * 10}
        )

        packet = context_dna_manager.compress_for_target_model(
            dna=dna,
            target_model="gemma-4-e4b",
            target_context_limit=8192
        )

        self.assertIsInstance(packet, HandoverPacket)
        self.assertTrue(packet.compression_ratio >= 1.0)
        self.assertTrue(packet.constraint_retention_rate >= 50.0)
        self.assertTrue(packet.continuity_valid)
        self.assertIn("OBJECTIVE", packet.compressed_prompt)
        self.assertIn("pipeline.py", packet.retained_artifacts)


if __name__ == "__main__":
    unittest.main()
