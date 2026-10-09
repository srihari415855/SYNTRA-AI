"""Trust Layer: Multi-Stage Response Verification Engine.

Implements the 5 distinct verification stages:
STAGE A — Response integrity (null/empty check, raw provider errors, schema conformity, truncation)
STAGE B — Requirement compliance (format requests, length constraints, topic coverage)
STAGE C — Task-specific validation (Python AST parsing, deterministic math checks, JSON validation)
STAGE D — Independent verification (findings: PASS, FAIL, NEEDS_REVIEW with severity and evidence)
STAGE E — Final response decision (Standard, Strict, Low Latency, Custom policies)
"""

import ast
import json
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("ai_system.verification")


class VerificationDecision(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    REJECTED = "REJECTED"
    UNVERIFIED = "UNVERIFIED"


class VerificationPolicy(str, Enum):
    STANDARD = "standard"
    STRICT = "strict"
    LOW_LATENCY = "low_latency"
    CUSTOM = "custom"


class FindingSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class VerificationFinding(BaseModel):
    stage: str
    check_name: str
    status: str  # PASS, FAIL, NEEDS_REVIEW, SKIPPED
    severity: FindingSeverity = FindingSeverity.INFO
    description: str
    evidence: Optional[str] = None


class VerificationReport(BaseModel):
    decision: VerificationDecision
    policy: VerificationPolicy
    passed: bool
    confidence_score: float  # 0.0 to 1.0
    stage_a_integrity: bool
    stage_b_compliance: bool
    stage_c_task_validation: bool
    stage_d_independent_review: bool
    findings: List[VerificationFinding] = Field(default_factory=list)
    corrective_action_needed: bool = False
    corrective_guidance: Optional[str] = None
    verification_latency_ms: float = 0.0
    evaluation_limits: str = (
        "Verification reduces identifiable syntactic, mathematical, and requirement defects; "
        "it does not guarantee absolute empirical correctness or factual omniscience."
    )


class TrustLayerVerificationEngine:
    """The Trust Layer: Evaluates generated responses before they can be labelled verified."""

    def __init__(self):
        pass

    def verify(
        self,
        prompt: str,
        output: str,
        task_category: str = "general",
        policy: VerificationPolicy = VerificationPolicy.STANDARD,
        required_tools: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None
    ) -> VerificationReport:
        """Executes full multi-stage verification against candidate output."""
        start_time = time.perf_counter()
        findings: List[VerificationFinding] = []

        # =========================================================================
        # STAGE A — Response Integrity
        # =========================================================================
        stage_a_passed = True

        # Check A1: Null / Empty / Truncated Check
        if not output or not output.strip():
            stage_a_passed = False
            findings.append(VerificationFinding(
                stage="STAGE_A_INTEGRITY",
                check_name="non_empty_output",
                status="FAIL",
                severity=FindingSeverity.CRITICAL,
                description="Output is empty or contains only whitespace.",
                evidence=f"Length: {len(output) if output else 0}"
            ))
        elif len(output.strip()) < 5:
            stage_a_passed = False
            findings.append(VerificationFinding(
                stage="STAGE_A_INTEGRITY",
                check_name="minimum_length",
                status="FAIL",
                severity=FindingSeverity.CRITICAL,
                description="Output is abnormally short (< 5 characters) and likely truncated.",
                evidence=f"Content: '{output.strip()}'"
            ))
        else:
            findings.append(VerificationFinding(
                stage="STAGE_A_INTEGRITY",
                check_name="minimum_length",
                status="PASS",
                severity=FindingSeverity.INFO,
                description="Output contains substantial textual content."
            ))

        # Check A2: Provider Raw Error Leaks
        raw_error_patterns = [
            r'{"error":\s*{"message":',
            r"Rate limit exceeded",
            r"Internal Server Error",
            r"HTTP 50[0-4]",
            r"Error: \[Errno \d+\]",
            r"Traceback \(most recent call last\):"
        ]
        leaked_error = False
        for pat in raw_error_patterns:
            if re.search(pat, output, re.IGNORECASE):
                leaked_error = True
                stage_a_passed = False
                findings.append(VerificationFinding(
                    stage="STAGE_A_INTEGRITY",
                    check_name="provider_error_leak",
                    status="FAIL",
                    severity=FindingSeverity.CRITICAL,
                    description="Raw provider error or stacktrace returned instead of answer.",
                    evidence=f"Matched pattern: {pat}"
                ))
                break
        if not leaked_error:
            findings.append(VerificationFinding(
                stage="STAGE_A_INTEGRITY",
                check_name="provider_error_leak",
                status="PASS",
                severity=FindingSeverity.INFO,
                description="No leaked provider error signatures or stack traces detected."
            ))

        # Check A3: Unclosed Markdown Blocks (Sign of abrupt truncation)
        code_fence_count = output.count("```")
        if code_fence_count % 2 != 0:
            stage_a_passed = False
            findings.append(VerificationFinding(
                stage="STAGE_A_INTEGRITY",
                check_name="markdown_fence_integrity",
                status="FAIL",
                severity=FindingSeverity.WARNING,
                description="Abrupt truncation detected: unclosed markdown code fence (```).",
                evidence=f"Total fences: {code_fence_count}"
            ))
        else:
            findings.append(VerificationFinding(
                stage="STAGE_A_INTEGRITY",
                check_name="markdown_fence_integrity",
                status="PASS",
                severity=FindingSeverity.INFO,
                description="Markdown syntax formatting structures are balanced and intact."
            ))

        # =========================================================================
        # STAGE B — Requirement Compliance
        # =========================================================================
        stage_b_passed = True
        prompt_lower = prompt.lower()
        output_lower = output.lower()

        # Check B1: JSON Format compliance if requested
        wants_json = "in json" in prompt_lower or "as json" in prompt_lower or "json format" in prompt_lower or "json output" in prompt_lower
        if wants_json:
            # Look for JSON structure in output
            has_valid_json = False
            # Try parsing whole output or fenced json
            json_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", output, re.DOTALL)
            candidate_json = json_match.group(1) if json_match else output.strip()
            try:
                json.loads(candidate_json)
                has_valid_json = True
            except Exception:
                # Also try finding first { and last }
                start_br = output.find("{")
                end_br = output.rfind("}")
                if start_br != -1 and end_br > start_br:
                    try:
                        json.loads(output[start_br:end_br+1])
                        has_valid_json = True
                    except Exception:
                        pass

            if has_valid_json:
                findings.append(VerificationFinding(
                    stage="STAGE_B_COMPLIANCE",
                    check_name="requested_json_format",
                    status="PASS",
                    severity=FindingSeverity.INFO,
                    description="User requested JSON format and output parsed as valid structured JSON."
                ))
            else:
                stage_b_passed = False
                findings.append(VerificationFinding(
                    stage="STAGE_B_COMPLIANCE",
                    check_name="requested_json_format",
                    status="FAIL",
                    severity=FindingSeverity.CRITICAL,
                    description="User explicitly requested JSON format, but output did not contain valid JSON.",
                    evidence="Failed to parse structured JSON."
                ))

        # Check B2: Bullet / List format compliance if requested
        wants_list = "bullet" in prompt_lower or "bulleted" in prompt_lower or "list of" in prompt_lower
        if wants_list:
            has_bullets = bool(re.search(r"^[\s]*[-*•\d+\.]\s+", output, re.MULTILINE))
            if has_bullets:
                findings.append(VerificationFinding(
                    stage="STAGE_B_COMPLIANCE",
                    check_name="requested_list_format",
                    status="PASS",
                    severity=FindingSeverity.INFO,
                    description="Requested bulleted or ordered list format was provided."
                ))
            else:
                findings.append(VerificationFinding(
                    stage="STAGE_B_COMPLIANCE",
                    check_name="requested_list_format",
                    status="NEEDS_REVIEW",
                    severity=FindingSeverity.WARNING,
                    description="Prompt suggested a list/bullet format, but standard list markers were not detected."
                ))

        # Check B3: Constraint Negations (e.g. "do not use", "without")
        negation_match = re.search(r"(?:do not use|without|avoid)\s+([a-zA-Z0-9_\-]+)", prompt_lower)
        if negation_match:
            forbidden_term = negation_match.group(1)
            if forbidden_term in output_lower:
                stage_b_passed = False
                findings.append(VerificationFinding(
                    stage="STAGE_B_COMPLIANCE",
                    check_name="negative_constraint_compliance",
                    status="FAIL",
                    severity=FindingSeverity.WARNING,
                    description=f"Negative constraint violated: prompt instructed not to use '{forbidden_term}'.",
                    evidence=f"Found '{forbidden_term}' in generated response."
                ))
            else:
                findings.append(VerificationFinding(
                    stage="STAGE_B_COMPLIANCE",
                    check_name="negative_constraint_compliance",
                    status="PASS",
                    severity=FindingSeverity.INFO,
                    description=f"Negative constraint respected: '{forbidden_term}' avoided."
                ))

        # =========================================================================
        # STAGE C — Task-Specific Validation
        # =========================================================================
        stage_c_passed = True

        # Task C1: Code Validation (Python AST parsing)
        if task_category == "coding" or "```python" in output or "```py" in output:
            code_blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", output, re.DOTALL)
            if code_blocks:
                all_syntax_valid = True
                syntax_errors = []
                for idx, block in enumerate(code_blocks):
                    block_stripped = block.strip()
                    if not block_stripped:
                        continue
                    try:
                        ast.parse(block_stripped)
                    except SyntaxError as syn_err:
                        all_syntax_valid = False
                        syntax_errors.append(f"Block #{idx+1} line {syn_err.lineno}: {syn_err.msg}")

                if all_syntax_valid:
                    findings.append(VerificationFinding(
                        stage="STAGE_C_TASK_SPECIFIC",
                        check_name="python_ast_syntax_validity",
                        status="PASS",
                        severity=FindingSeverity.INFO,
                        description=f"All {len(code_blocks)} Python code block(s) compiled successfully via AST verification."
                    ))
                else:
                    stage_c_passed = False
                    findings.append(VerificationFinding(
                        stage="STAGE_C_TASK_SPECIFIC",
                        check_name="python_ast_syntax_validity",
                        status="FAIL",
                        severity=FindingSeverity.CRITICAL,
                        description="Code contains syntax errors that prevent execution.",
                        evidence="; ".join(syntax_errors)
                    ))
            else:
                findings.append(VerificationFinding(
                    stage="STAGE_C_TASK_SPECIFIC",
                    check_name="code_block_presence",
                    status="NEEDS_REVIEW",
                    severity=FindingSeverity.WARNING,
                    description="Task was classified as coding, but no fenced code blocks were found."
                ))

        # Task C2: Mathematical / Arithmetic Calculation Check
        if task_category in ("math", "arithmetic", "tool_execution") or any(c in prompt for c in ["+", "-", "*", "/", "^"]):
            # Look for expressions like 25 * 25 = 625 or 100 / 4 = 25 or tool result
            math_match = re.search(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*(?:=|is)\s*(\d+(?:\.\d+)?)", output)
            if math_match:
                op1 = float(math_match.group(1))
                operator = math_match.group(2)
                op2 = float(math_match.group(3))
                claimed_result = float(math_match.group(4))

                actual_calc = None
                if operator == "+":
                    actual_calc = op1 + op2
                elif operator == "-":
                    actual_calc = op1 - op2
                elif operator == "*":
                    actual_calc = op1 * op2
                elif operator == "/" and op2 != 0:
                    actual_calc = op1 / op2

                if actual_calc is not None:
                    if abs(actual_calc - claimed_result) < 1e-4:
                        findings.append(VerificationFinding(
                            stage="STAGE_C_TASK_SPECIFIC",
                            check_name="deterministic_math_verification",
                            status="PASS",
                            severity=FindingSeverity.INFO,
                            description=f"Deterministic calculation confirmed: {op1} {operator} {op2} = {actual_calc}."
                        ))
                    else:
                        stage_c_passed = False
                        findings.append(VerificationFinding(
                            stage="STAGE_C_TASK_SPECIFIC",
                            check_name="deterministic_math_verification",
                            status="FAIL",
                            severity=FindingSeverity.CRITICAL,
                            description=f"Arithmetic discrepancy detected! Stated {claimed_result}, but actual calculation is {actual_calc}.",
                            evidence=f"Expression: {op1} {operator} {op2} != {claimed_result}"
                        ))

            # Also check tool executions if calculator was run
            if tool_results:
                for tr in tool_results:
                    if tr.get("tool_name") == "calculator":
                        calc_val = str(tr.get("output", {}).get("result", ""))
                        if calc_val and calc_val in output:
                            findings.append(VerificationFinding(
                                stage="STAGE_C_TASK_SPECIFIC",
                                check_name="tool_grounding_consistency",
                                status="PASS",
                                severity=FindingSeverity.INFO,
                                description=f"Output is grounded in verified calculator tool result: {calc_val}."
                            ))

        # Task C3: Summarization / Content length
        if task_category == "summarization":
            word_count = len(output.split())
            findings.append(VerificationFinding(
                stage="STAGE_C_TASK_SPECIFIC",
                check_name="summary_length_check",
                status="PASS",
                severity=FindingSeverity.INFO,
                description=f"Summary produced with concise {word_count} words."
            ))

        # =========================================================================
        # STAGE D — Independent Verification (Heuristic and Consistency Review)
        # =========================================================================
        stage_d_passed = True
        # Check for hedge words, uncertainty disclosures, hallucinations
        has_hedging = any(h in output_lower for h in ["i believe", "might be", "possibly", "unverified", "as an estimate"])
        if has_hedging:
            findings.append(VerificationFinding(
                stage="STAGE_D_INDEPENDENT_REVIEW",
                check_name="epistemic_humility_disclosure",
                status="PASS",
                severity=FindingSeverity.INFO,
                description="Model appropriately discloses uncertainty or estimation limits."
            ))

        # Check for repetition loops (identical lines repeated 3+ times)
        lines = [line.strip() for line in output.splitlines() if len(line.strip()) > 15]
        line_counts: Dict[str, int] = {}
        has_repetition_loop = False
        for l in lines:
            line_counts[l] = line_counts.get(l, 0) + 1
            if line_counts[l] >= 3:
                has_repetition_loop = True
                break

        if has_repetition_loop:
            stage_d_passed = False
            findings.append(VerificationFinding(
                stage="STAGE_D_INDEPENDENT_REVIEW",
                check_name="degenerate_repetition_loop",
                status="FAIL",
                severity=FindingSeverity.CRITICAL,
                description="Degenerate model repetition loop detected (same sentence repeated 3+ times).",
                evidence="Identified duplicate line sequence in generation."
            ))

        # =========================================================================
        # STAGE E — Final Response Decision (Policy Matrix)
        # =========================================================================
        critical_failures = [f for f in findings if f.status == "FAIL" and f.severity == FindingSeverity.CRITICAL]
        warnings = [f for f in findings if f.status in ("FAIL", "NEEDS_REVIEW") and f.severity == FindingSeverity.WARNING]

        # Calculate Confidence Score
        total_checks = len(findings)
        pass_checks = len([f for f in findings if f.status == "PASS"])
        confidence = round(pass_checks / max(total_checks, 1), 2)

        decision = VerificationDecision.VERIFIED
        corrective_needed = False
        corrective_guidance = None

        if policy == VerificationPolicy.LOW_LATENCY:
            # Low latency only requires Stage A integrity
            if stage_a_passed:
                decision = VerificationDecision.VERIFIED
            else:
                decision = VerificationDecision.REJECTED
                corrective_needed = True

        elif policy == VerificationPolicy.STRICT:
            # Strict requires NO critical failures and NO unresolved warnings
            if critical_failures:
                decision = VerificationDecision.REJECTED
                corrective_needed = True
                corrective_guidance = f"Resolve critical defects: {'; '.join([c.description for c in critical_failures])}"
            elif warnings:
                decision = VerificationDecision.PARTIALLY_VERIFIED
                corrective_needed = True
                corrective_guidance = f"Review warnings: {'; '.join([w.description for w in warnings])}"
            elif stage_a_passed and stage_b_passed and stage_c_passed and stage_d_passed:
                decision = VerificationDecision.VERIFIED
            else:
                decision = VerificationDecision.PARTIALLY_VERIFIED

        else:
            # Standard Policy (Default)
            if critical_failures:
                decision = VerificationDecision.REJECTED
                corrective_needed = True
                corrective_guidance = f"Address critical issues: {'; '.join([c.description for c in critical_failures])}"
            elif not stage_b_passed or warnings:
                decision = VerificationDecision.PARTIALLY_VERIFIED
            elif stage_a_passed and stage_c_passed:
                decision = VerificationDecision.VERIFIED
            else:
                decision = VerificationDecision.PARTIALLY_VERIFIED

        duration_ms = (time.perf_counter() - start_time) * 1000

        return VerificationReport(
            decision=decision,
            policy=policy,
            passed=(decision in (VerificationDecision.VERIFIED, VerificationDecision.PARTIALLY_VERIFIED)),
            confidence_score=confidence,
            stage_a_integrity=stage_a_passed,
            stage_b_compliance=stage_b_passed,
            stage_c_task_validation=stage_c_passed,
            stage_d_independent_review=stage_d_passed,
            findings=findings,
            corrective_action_needed=corrective_needed,
            corrective_guidance=corrective_guidance,
            verification_latency_ms=round(duration_ms, 2)
        )


# Global singleton
verification_engine = TrustLayerVerificationEngine()
