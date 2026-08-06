import os
import time

import agentguard
from dotenv import load_dotenv
from google import genai
from google.genai import types
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


# ============================================================
# Configuration
# ============================================================

load_dotenv(override=True)

# Capture Gemini prompts and responses.
os.environ["AGENTGUARD_CAPTURE_CONTENT"] = "true"

agentguard.init(
    public_key=os.getenv("AGENTGUARD_PUBLIC_KEY"),
    secret_key=os.getenv("AGENTGUARD_SECRET_KEY"),
    base_url=os.getenv("AGENTGUARD_BASE_URL"),
    project_id=os.getenv("AGENTGUARD_PROJECT_ID"),
    guardrails="auto",
)

# Native Google Gemini SDK client.
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)

# This tracer uses the provider configured by agentguard.init().
tracer = trace.get_tracer("agentguard-observability-tests")

CATEGORY_FAILURES = []


# ============================================================
# Common helpers
# ============================================================

def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def run_gemini_test(
    *,
    test_case,
    model="gemini-2.5-flash",
    feature="classify-intent",
    tenant="acme-corp",
    prompt="Reply with TEST OK",
):
    """Make a real Gemini SDK request captured by AgentGuard."""

    print(f"\n[START] {test_case}")
    print(f"Model   : {model}")
    print(f"Feature : {feature}")
    print(f"Tenant  : {tenant}")

    try:
        with agentguard.policy(
            disable=["prompt-injection", "toxic-content"],
            user_id=tenant,
            session_id=f"session-{test_case}",
            feature=feature,
            metadata={
                "tenant": tenant,
                "test_case": test_case,
                "sdk": "google-genai",
            },
        ):
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=50,
                ),
            )

        content = response.text

        print(f"[PASS] {test_case}")
        print("Response:", content)

        return True

    except Exception as error:
        failure = (
            f"{test_case}: "
            f"{type(error).__name__}: {error}"
        )

        CATEGORY_FAILURES.append(failure)

        print(f"[FAIL] {test_case}")
        print("Error type:", type(error).__name__)
        print("Error:", error)

        return False


def create_generation_observation(
    *,
    name,
    model,
    feature,
    tenant,
    input_tokens,
    output_tokens,
    duration=0,
):
    """Create generation data with deterministic usage and duration."""

    print(f"\n[START] {name}")

    with tracer.start_as_current_span(name) as span:

        span.set_attribute("gen_ai.system", "google")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.response.model", model)

        span.set_attribute(
            "gen_ai.usage.input_tokens",
            input_tokens,
        )
        span.set_attribute(
            "gen_ai.usage.output_tokens",
            output_tokens,
        )
        span.set_attribute(
            "gen_ai.usage.total_tokens",
            input_tokens + output_tokens,
        )

        if duration > 0:
            time.sleep(duration)

        span.set_status(Status(StatusCode.OK))

    print(f"[PASS] {name}")

    return True


def create_error_observation(
    *,
    name,
    message,
    feature="error-testing",
    tenant="error-test-tenant",
    tool_name=None,
):
    """Create an intentional error observation."""

    print(f"\n[START] {name}")
    print(f"Expected error: {message}")

    with tracer.start_as_current_span(name) as span:
        if tool_name:
            span.set_attribute(
                "gen_ai.tool.name",
                tool_name,
            )

        error = RuntimeError(message)

        span.record_exception(error)
        span.set_attribute(
            "error.type",
            type(error).__name__,
        )
        span.set_attribute(
            "error.message",
            message,
        )
        span.set_status(
            Status(StatusCode.ERROR, message),
        )

    print(f"[PASS] Expected error telemetry created: {name}")

    return True


# ============================================================
# 1. Cost by Model
# ============================================================

def test_cost_by_model():
    print_section("1. COST BY MODEL")

    cases = [
        {
            "test_case": "cost-model-gemini-2.5-flash",
            "model": os.getenv(
                "TEST_MODEL_1",
                "gemini-2.5-flash",
            ),
        },
        {
            "test_case": "cost-model-gemini-2.5-flash-lite",
            "model": os.getenv(
                "TEST_MODEL_2",
                "gemini-2.5-flash-lite",
            ),
        },
        {
            "test_case": "cost-model-gemini-2.5-pro",
            "model": os.getenv(
                "TEST_MODEL_3",
                "gemini-2.5-pro",
            ),
        },
    ]

    for case in cases:
        run_gemini_test(
            test_case=case["test_case"],
            model=case["model"],
            feature="model-cost-test",
            tenant="model-test-tenant",
            prompt="Reply with exactly: MODEL TEST OK",
        )


# ============================================================
# 2. Cost by Feature
# ============================================================

def test_cost_by_feature():
    print_section("2. COST BY FEATURE")

    cases = [
        {
            "feature": "classify-intent",
            "input_tokens": 500,
            "output_tokens": 50,
        },
        {
            "feature": "summarize-document",
            "input_tokens": 1500,
            "output_tokens": 200,
        },
        {
            "feature": "customer-support",
            "input_tokens": 800,
            "output_tokens": 120,
        },
    ]

    for case in cases:
        create_generation_observation(
            name=case["feature"],
            model="gemini-2.5-flash",
            feature=case["feature"],
            tenant="feature-test-tenant",
            input_tokens=case["input_tokens"],
            output_tokens=case["output_tokens"],
        )


# ============================================================
# 3. Cost by Tenant
# ============================================================

def test_cost_by_tenant():
    print_section("3. COST BY TENANT")

    tenants = [
        "actaclad-corp",
        "aanseaa-inc",
        "agentguard-ltd",
    ]

    for tenant in tenants:
        run_gemini_test(
            test_case=f"cost-tenant-{tenant}",
            model="gemini-2.5-flash",
            feature="customer-support",
            tenant=tenant,
            prompt=f"Reply with exactly: {tenant.upper()} OK",
        )


# ============================================================
# 4. Trace Errors
# ============================================================

def test_trace_errors():
    print_section("4. TRACE ERRORS")

    cases = [
        {
            "name": "trace-error-validation",
            "message": "Customer ID is missing",
        },
        {
            "name": "trace-error-timeout",
            "message": "Payment provider timed out",
        },
        {
            "name": "trace-error-dependency",
            "message": "Inventory service is unavailable",
        },
    ]

    for case in cases:
        create_error_observation(
            name=case["name"],
            message=case["message"],
            feature="trace-error-testing",
            tenant="trace-error-tenant",
        )


# ============================================================
# 5. Span Observations
# ============================================================

def test_spans():
    print_section("5. SPAN OBSERVATIONS")

    cases = [
        {
            "name": "retrieve-customer",
            "feature": "customer-lookup",
            "duration": 0.10,
        },
        {
            "name": "build-prompt",
            "feature": "prompt-builder",
            "duration": 0.20,
        },
        {
            "name": "parse-response",
            "feature": "response-parser",
            "duration": 0.15,
        },
    ]

    for case in cases:
        print(f"\n[START] {case['name']}")

        with tracer.start_as_current_span(
            case["name"],
        ) as span:
            span.set_attribute(
                "agentguard.operation.result",
                "success",
            )

            time.sleep(case["duration"])
            span.set_status(Status(StatusCode.OK))

        print(f"[PASS] {case['name']}")


# ============================================================
# 6. Tool Observations
# ============================================================

def test_tools():
    print_section("6. TOOL OBSERVATIONS")

    cases = [
        {
            "span_name": "weather-search",
            "tool_name": "weather_search",
            "arguments": '{"city": "Chennai"}',
            "result": (
                '{"temperature": 31, "unit": "celsius"}'
            ),
        },
        {
            "span_name": "order-lookup",
            "tool_name": "order_lookup",
            "arguments": '{"order_id": "ORD-1001"}',
            "result": '{"status": "shipped"}',
        },
        {
            "span_name": "calculator",
            "tool_name": "calculator",
            "arguments": '{"expression": "125 * 1.18"}',
            "result": '{"value": 147.5}',
        },
    ]

    for case in cases:
        print(f"\n[START] {case['span_name']}")

        with tracer.start_as_current_span(
            case["span_name"],
        ) as span:
            span.set_attribute(
                "gen_ai.tool.name",
                case["tool_name"],
            )
            span.set_attribute(
                "gen_ai.tool.call.arguments",
                case["arguments"],
            )
            span.set_attribute(
                "gen_ai.tool.call.result",
                case["result"],
            )

            time.sleep(0.10)
            span.set_status(Status(StatusCode.OK))

        print(f"[PASS] {case['span_name']}")


# ============================================================
# 7. Error Observations
# ============================================================

def test_error_observations():
    print_section("7. ERROR OBSERVATIONS")

    with tracer.start_as_current_span(
        "error-observation-test-suite",
    ) as parent_span:
        create_error_observation(
            name="json-parser-error",
            message="Invalid JSON returned by the model",
        )

        create_error_observation(
            name="tool-execution-error",
            message="Tool returned HTTP 500",
            tool_name="order_lookup",
        )

        create_error_observation(
            name="output-validation-error",
            message="Required output field 'answer' is missing",
        )

        parent_span.set_status(Status(StatusCode.OK))

    print("\n[PASS] Three child error observations created")


# ============================================================
# 8. Latency Observations
# ============================================================

def test_latency():
    print_section("8. LATENCY OBSERVATIONS")

    cases = [
        {
            "name": "latency-fast-200ms",
            "duration": 0.20,
        },
        {
            "name": "latency-medium-800ms",
            "duration": 0.80,
        },
        {
            "name": "latency-slow-1500ms",
            "duration": 1.50,
        },
    ]

    for case in cases:
        print(
            f"\n[START] {case['name']} "
            f"(expected {case['duration']} seconds)"
        )

        create_generation_observation(
            name=case["name"],
            model="gemini-2.5-flash",
            feature="latency-test",
            tenant="latency-test-tenant",
            input_tokens=0,
            output_tokens=0,
            duration=case["duration"],
        )


# ============================================================
# Category runner
# ============================================================

def run_category(category_name, test_function):
    failures_before = len(CATEGORY_FAILURES)

    try:
        test_function()

    except Exception as error:
        failure = (
            f"{category_name}: "
            f"{type(error).__name__}: {error}"
        )

        CATEGORY_FAILURES.append(failure)

        print(f"\n[FAIL] {category_name}")
        print("Error type:", type(error).__name__)
        print("Error:", error)

    failures_after = len(CATEGORY_FAILURES)

    return {
        "name": category_name,
        "passed": failures_after == failures_before,
    }


# ============================================================
# Main
# ============================================================

def main():
    print_section("AGENTGUARD OBSERVABILITY TESTS STARTED")

    categories = [
        ("Cost by model", test_cost_by_model),
        ("Cost by feature", test_cost_by_feature),
        ("Cost by tenant", test_cost_by_tenant),
        ("Trace errors", test_trace_errors),
        ("Span observations", test_spans),
        ("Tool observations", test_tools),
        ("Error observations", test_error_observations),
        ("Latency observations", test_latency),
    ]

    results = []

    for category_name, test_function in categories:
        result = run_category(
            category_name,
            test_function,
        )
        results.append(result)

    print_section("FLUSHING OBSERVABILITY DATA")

    try:
        agentguard.flush()
        print("[PASS] AgentGuard flush completed")

    except Exception as error:
        CATEGORY_FAILURES.append(
            f"AgentGuard flush: {error}"
        )
        print(f"[FAIL] AgentGuard flush: {error}")

    passed_count = sum(
        1 for result in results if result["passed"]
    )
    failed_count = len(results) - passed_count

    print("\n")
    print("=" * 65)
    print("OBSERVABILITY TEST SUMMARY")
    print("=" * 65)

    for result in results:
        status = (
            "PASS"
            if result["passed"]
            else "FAILED"
        )

        print(
            f"OBS: {result['name']:<25}: {status}"
        )

    print("-" * 65)
    print(f"TOTAL  : {len(results)}")
    print(f"PASS   : {passed_count}")
    print(f"FAILED : {failed_count}")
    print("=" * 65)

    if CATEGORY_FAILURES:
        print("\nFAILURE DETAILS")

        for number, failure in enumerate(
            CATEGORY_FAILURES,
            start=1,
        ):
            print(f"{number}. {failure}")

    if failed_count == 0:
        print("\n[DONE] All Observability tests passed.")
    else:
        print(
            f"\n[DONE] {failed_count} Observability "
            "category test(s) failed."
        )

    print(
        "\nCheck AgentGuard Observability using "
        "'Last 24 hours' or 'Last 7 days'."
    )


if __name__ == "__main__":
    main()
