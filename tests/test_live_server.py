"""Live E2E Verification of Running SYNTRA AI / OpenPilot Server."""

import json
import httpx


def test_live_server():
    base_url = "http://localhost:8000"
    client = httpx.Client(base_url=base_url, timeout=15.0)

    print("1. Testing Health Endpoint...")
    h = client.get("/health")
    assert h.status_code == 200, f"Health failed: {h.text}"
    print("[OK] Health Check:", h.json())

    print("\n2. Testing UI HTML Serving...")
    index_res = client.get("/")
    assert index_res.status_code == 200
    assert "SYNTRA AI" in index_res.text
    assert "Jury Failover Demo" in index_res.text
    assert "trust-layer-card" in index_res.text
    assert "context-dna-card" in index_res.text
    assert "recovery-controls-card" in index_res.text
    print("[OK] UI Serving verified (HTML contains Trust Layer, Context DNA, and Jury Demo modal).")

    print("\n3. Testing Demo Auth...")
    auth_res = client.post("/api/v1/auth/demo")
    assert auth_res.status_code == 200
    token = auth_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[OK] Authenticated as Demo Pilot:", auth_res.json()["name"])

    print("\n4. Testing Request Execution with Trust Layer & Context DNA...")
    exec_payload = {
        "prompt": "calculate 25 * (100 / 4) + 15",
        "ranking_mode": "best_value"
    }
    exec_res = client.post("/api/v1/execute", json=exec_payload, headers=headers)
    assert exec_res.status_code == 200
    exec_data = exec_res.json()
    assert "verification" in exec_data, "Missing verification in execution result"
    assert "context_dna" in exec_data, "Missing context_dna in execution result"
    assert "trace" in exec_data, "Missing trace in execution result"
    print("[OK] Execution Output:", exec_data["final_output"])
    print("[OK] Trust Layer Decision:", exec_data["verification"]["decision"], f"(Confidence: {exec_data['verification']['confidence_score']*100}%)")
    print("[OK] Context DNA Goal:", exec_data["context_dna"]["objective"])
    print("[OK] Trace Events Recorded:", len(exec_data["trace"]["events"]), f"(Latency: {exec_data['trace']['total_elapsed_ms']}ms)")

    print("\n5. Testing Jury Failover Demo Endpoint...")
    jury_payload = {
        "prompt": "Implement a Python pipeline to process sensor readings with statistical aggregations in JSON format.",
        "source_model": "gpt-4o",
        "simulated_failure": "Error 402: You have exceeded your current quota / billing limit reached",
        "prefer_free_only": True
    }
    jury_res = client.post("/api/v1/demo/simulate-failure-migration", json=jury_payload, headers=headers)
    assert jury_res.status_code == 200
    jury_data = jury_res.json()
    assert jury_data["success"] is True
    assert jury_data["source_model"] == "gpt-4o"
    assert jury_data["target_provider"] != "OpenAI", "Circuit breaker should not reuse failed provider"
    assert jury_data["circuit_breaker_active"] is True
    print("[OK] Failover Trigger:", jury_data["trigger_reason"])
    print("[OK] Failure Classified As:", jury_data["failure_category"])
    print(f"[OK] Circuit Breaker Rerouted: {jury_data['source_model']} ({jury_data['source_provider']}) -> {jury_data['target_model']} ({jury_data['target_provider']})")
    print(f"[OK] Token Compression: {jury_data['compression_ratio']}x reduction, {jury_data['retention_rate']}% constraints retained")
    print("[OK] Continuation Output Verified:", jury_data["verification_report"]["decision"])
    print(f"[OK] Total Handover Latency: {jury_data['total_latency_ms']}ms across {len(jury_data['execution_trace']['events'])} trace events")

    print("\n6. Testing Direct Trust Layer Verification Endpoint...")
    v_res = client.post("/api/v1/verify", json={
        "prompt": "Write Python code",
        "output": "```python\ndef compute(n: int) -> int:\n    return n * 2\n```",
        "task_category": "coding",
        "policy": "strict"
    })
    assert v_res.status_code == 200
    assert v_res.json()["decision"] == "VERIFIED"
    print("[OK] Direct Verification Report:", v_res.json()["decision"])

    print("\n7. Testing Feedback & Quality Recovery Endpoint...")
    fb_res = client.post("/api/v1/feedback", json={
        "session_id": exec_data["session_id"],
        "prompt": "Refine output",
        "current_output": exec_data["final_output"],
        "feedback_type": "verify"
    }, headers=headers)
    assert fb_res.status_code == 200
    assert "verification_report" in fb_res.json()
    print("[OK] Feedback Verification Recovery:", fb_res.json()["verification_report"]["decision"])

    print("\n=======================================================")
    print("[SUCCESS] ALL 7 LIVE SERVER INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=======================================================")


if __name__ == "__main__":
    test_live_server()
