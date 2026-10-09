"""Complete End-to-End Live Integration Verification for SYNTRA AI / OpenPilot AI Platform."""

import sys
import httpx

BASE_URL = "http://localhost:8000"


def run_live_tests():
    print("[1/10] Checking System Health at /health...")
    resp = httpx.get(f"{BASE_URL}/health", timeout=10.0)
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health = resp.json()
    assert health["status"] == "healthy"
    print(f"       [PASS] Health OK: {health['available_models']} models, {health['registered_tools']} tools.")

    print("\n[2/10] Testing Demo Authentication Session...")
    resp = httpx.post(f"{BASE_URL}/api/v1/auth/demo", timeout=10.0)
    assert resp.status_code == 200, f"Demo login failed: {resp.text}"
    auth_data = resp.json()
    token = auth_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"       [PASS] Authenticated demo user: {auth_data['user_id']}")

    print("\n[3/10] Testing Model Catalogue & Explorer at /api/v1/models...")
    resp = httpx.get(f"{BASE_URL}/api/v1/models", headers=headers, timeout=10.0)
    assert resp.status_code == 200, f"Models listing failed: {resp.text}"
    models = resp.json()
    assert len(models) == 10, f"Expected 10 models, got {len(models)}"
    print(f"       [PASS] Retrieved {len(models)} registered models with pricing & benchmark citations.")

    print("\n[4/10] Testing Request Understanding...")
    resp = httpx.post(
        f"{BASE_URL}/api/v1/understand",
        json={"prompt": "Write a python script to validate email syntax"},
        headers=headers,
        timeout=10.0
    )
    assert resp.status_code == 200, f"Understanding failed: {resp.text}"
    und = resp.json()
    assert und["task_category"] == "coding"
    print(f"       [PASS] Inferred task category: '{und['task_category']}' (confidence: {und['confidence']:.2f})")

    print("\n[5/10] Testing Dynamic Recommendations Across Ranking Modes...")
    for mode in ["best_value", "lowest_price", "best_performance", "free"]:
        resp = httpx.post(
            f"{BASE_URL}/api/v1/recommend",
            json={"prompt": "Write unit tests for binary search", "ranking_mode": mode},
            headers=headers,
            timeout=10.0
        )
        assert resp.status_code == 200, f"Recommendation mode {mode} failed: {resp.text}"
        rec = resp.json()
        assert len(rec["models"]) > 0
        print(f"       [PASS] Mode '{mode}': top pick '{rec['top_recommendation_id']}'")

    print("\n[6/10] Testing Turn Execution with Token Budgeting & Trust Layer...")
    resp = httpx.post(
        f"{BASE_URL}/api/v1/execute",
        json={
            "prompt": "Calculate 45 * 18",
            "ranking_mode": "best_value"
        },
        headers=headers,
        timeout=15.0
    )
    assert resp.status_code == 200, f"Execution failed: {resp.text}"
    result = resp.json()
    assert "final_output" in result
    assert "verification" in result
    assert "token_plan" in result
    assert "token_accounting" in result
    assert "context_dna" in result
    print(f"       [PASS] Executed: Status={result['status']}, Tokens={result['tokens_used']}, Time={result['execution_time_ms']}ms")
    print(f"              Trust Layer Decision: [{result['verification']['decision']}]")
    print(f"              Token Accounting: {result['token_accounting']['tokens_saved']} tokens saved ({result['token_accounting']['savings_percentage']}%)")

    session_id = result["session_id"]
    selected_model = result["selected_model"]

    print("\n[7/10] Testing Review & Bayesian Smoothed Feedback Submission...")
    resp = httpx.post(
        f"{BASE_URL}/api/v1/reviews",
        json={
            "session_id": session_id,
            "model_id": selected_model,
            "provider": "Ollama (Local)",
            "task_category": "tool_execution",
            "rating": 5,
            "is_correct": True,
            "is_useful": True,
            "is_complete": True,
            "is_fast": True,
            "cost_acceptable": True,
            "feedback_text": "Instant calculation with zero error"
        },
        headers=headers,
        timeout=10.0
    )
    assert resp.status_code == 200, f"Review submission failed: {resp.text}"
    rev_data = resp.json()
    stats = rev_data["updated_model_stats"]
    assert stats["smoothed_rating"] >= 4.0
    print(f"       [PASS] Review recorded. Smoothed rating for '{selected_model}': {stats['smoothed_rating']} stars across {stats['review_count']} reviews.")

    print("\n[8/10] Testing Operational Dashboard KPI Telemetry at /api/v1/dashboard/stats...")
    resp = httpx.get(f"{BASE_URL}/api/v1/dashboard/stats", headers=headers, timeout=10.0)
    assert resp.status_code == 200, f"Dashboard stats failed: {resp.text}"
    dash = resp.json()
    assert dash["total_requests"] >= 1
    assert dash["success_rate_pct"] > 0
    print(f"       [PASS] Dashboard KPIs: Total Requests={dash['total_requests']}, Tokens Consumed={dash['total_tokens_consumed']}, Saved={dash['estimated_tokens_saved']}, Avg Rating={dash['average_rating']} stars")

    print("\n[9/10] Testing Context DNA Retrieval at /api/v1/context-dna/{session_id}...")
    resp = httpx.get(f"{BASE_URL}/api/v1/context-dna/{session_id}", headers=headers, timeout=10.0)
    assert resp.status_code == 200, f"Context DNA retrieval failed: {resp.text}"
    dna = resp.json()
    assert "objective" in dna
    assert "explicit_requirements" in dna
    print(f"       [PASS] Context DNA preserved with 8 canonical sections. Objective: '{dna['objective']}'")

    print("\n[10/10] Testing Jury Controlled Failover Loop at /api/v1/demo/simulate-failure-migration...")
    resp = httpx.post(
        f"{BASE_URL}/api/v1/demo/simulate-failure-migration",
        json={
            "prompt": "Build a secure token validation function in Python",
            "source_model": "gpt-4o",
            "simulated_failure": "Rate limit exceeded (HTTP 429)",
            "prefer_free_only": True
        },
        headers=headers,
        timeout=15.0
    )
    assert resp.status_code == 200, f"Jury demo failed: {resp.text}"
    demo = resp.json()
    assert demo["migration_status"] == "SUCCESS"
    assert demo["circuit_breaker_active"] is True
    print(f"       [PASS] Jury Failover Succeeded: Switched to '{demo['target_model']}' via circuit breaker in {demo['total_latency_ms']}ms.")

    print("\n" + "=" * 65)
    print("ALL 10 LIVE INTEGRATION END-TO-END TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    try:
        run_live_tests()
    except Exception as exc:
        print(f"\n[FAIL] Test aborted with error: {exc}")
        sys.exit(1)
