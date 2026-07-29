from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv(override=True)

# Configure content capture before importing AgentGuard.
os.environ.setdefault("AGENTGUARD_CAPTURE_CONTENT", "true")

import agentguard  # noqa: E402
import httpx  # noqa: E402
import openai  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.trace import Status, StatusCode  # noqa: E402


RUN_ID = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"

ENVIRONMENT = os.getenv(
    "AGENTGUARD_TEST_ENVIRONMENT",
    "observability-test",
)

TIMEOUT_SECONDS = int(
    os.getenv("AGENTGUARD_TEST_TIMEOUT", "120")
)

POLL_INTERVAL_SECONDS = 2.0

PUBLIC_KEY = os.getenv("AGENTGUARD_PUBLIC_KEY")
SECRET_KEY = os.getenv("AGENTGUARD_SECRET_KEY")
BASE_URL = os.getenv("AGENTGUARD_BASE_URL", "").rstrip("/")
PROJECT_ID = os.getenv("AGENTGUARD_PROJECT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_1 = os.getenv("TEST_MODEL_1", "gpt-4o-mini")
MODEL_2 = os.getenv("TEST_MODEL_2", "gpt-4.1-mini")
MODEL_3 = os.getenv("TEST_MODEL_3", "gpt-4.1-nano")


# ============================================================
# Exceptions
# ============================================================

class VerificationError(RuntimeError):
    """Raised when AgentGuard read-back verification fails."""


# ============================================================
# Validate configuration
# ============================================================

def require_configuration() -> None:
    required = {
        "AGENTGUARD_PUBLIC_KEY": PUBLIC_KEY,
        "AGENTGUARD_SECRET_KEY": SECRET_KEY,
        "AGENTGUARD_BASE_URL": BASE_URL,
        "AGENTGUARD_PROJECT_ID": PROJECT_ID,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise VerificationError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


require_configuration()


# ============================================================
# Initialize AgentGuard and OpenAI
# ============================================================

agentguard.init(
    public_key=PUBLIC_KEY,
    secret_key=SECRET_KEY,
    base_url=BASE_URL,
    project_id=PROJECT_ID,
    environment=ENVIRONMENT,
    guardrails="auto",
)

openai_client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
)

tracer = trace.get_tracer(
    "agentguard-observability-e2e-tests"
)

api_client = httpx.Client(
    base_url=BASE_URL,
    auth=httpx.BasicAuth(
        PUBLIC_KEY,
        SECRET_KEY,
    ),
    timeout=30,
    headers={
        "Accept": "application/json",
    },
)


# ============================================================
# Output helpers
# ============================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def fail(message: str) -> None:
    raise VerificationError(message)


def assert_true(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        fail(message)


# ============================================================
# OpenTelemetry helpers
# ============================================================

def span_ids(span: Any) -> Tuple[str, str]:
    context = span.get_span_context()

    trace_id = format(
        context.trace_id,
        "032x",
    )

    span_id = format(
        context.span_id,
        "016x",
    )

    return trace_id, span_id


def add_test_attributes(
    span: Any,
    *,
    test_case: str,
    feature: str,
    tenant: str,
) -> None:
    # Standard identity attributes used by AgentGuard.
    span.set_attribute(
        "user.id",
        tenant,
    )

    span.set_attribute(
        "session.id",
        f"session-{RUN_ID}-{test_case}",
    )

    # AgentGuard custom attributes.
    span.set_attribute(
        "agentguard.environment",
        ENVIRONMENT,
    )

    span.set_attribute(
        "agentguard.trace.name",
        test_case,
    )

    span.set_attribute(
        "agentguard.test.run_id",
        RUN_ID,
    )

    span.set_attribute(
        "agentguard.test.case",
        test_case,
    )

    span.set_attribute(
        "agentguard.feature",
        feature,
    )

    span.set_attribute(
        "agentguard.tenant",
        tenant,
    )


# ============================================================
# Public API helpers
# ============================================================

def validate_agentguard_authentication() -> None:
    response = api_client.get(
        "/api/public/traces",
        params={"limit": 1},
    )

    if response.status_code != 200:
        fail(
            "AgentGuard authentication failed: "
            f"HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    payload = response.json()

    if not isinstance(payload, dict):
        fail(
            "AgentGuard traces API returned "
            "an unexpected response"
        )

    if "data" not in payload:
        fail(
            "AgentGuard traces API response "
            "does not contain data"
        )

    print(
        "[PASS] AgentGuard API authentication verified"
    )


def request_json(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = api_client.get(
        path,
        params=params,
    )

    if response.status_code != 200:
        raise VerificationError(
            f"GET {path} failed with "
            f"HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    payload = response.json()

    if not isinstance(payload, dict):
        raise VerificationError(
            f"GET {path} returned an invalid response"
        )

    return payload


def wait_for(
    description: str,
    reader: Callable[[], Optional[Any]],
) -> Any:
    deadline = (
        time.monotonic() + TIMEOUT_SECONDS
    )

    last_error: Optional[str] = None

    while time.monotonic() < deadline:
        try:
            result = reader()

            if result is not None:
                return result

        except Exception as error:
            last_error = str(error)

        time.sleep(POLL_INTERVAL_SECONDS)

    message = (
        f"Timed out after {TIMEOUT_SECONDS}s "
        f"waiting for {description}"
    )

    if last_error:
        message += f". Last error: {last_error}"

    raise VerificationError(message)


def wait_for_observation(
    observation_id: str,
    predicate: Optional[
        Callable[[Dict[str, Any]], bool]
    ] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    def read() -> Optional[Dict[str, Any]]:
        response = api_client.get(
            f"/api/public/observations/"
            f"{observation_id}"
        )

        if response.status_code == 404:
            return None

        if response.status_code != 200:
            raise VerificationError(
                "Observation lookup returned "
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        observation = response.json()

        if not isinstance(observation, dict):
            return None

        if (
            predicate is not None
            and not predicate(observation)
        ):
            return None

        return observation

    return wait_for(
        description
        or f"observation {observation_id}",
        read,
    )


def wait_for_trace(
    trace_id: str,
    predicate: Optional[
        Callable[[Dict[str, Any]], bool]
    ] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    def read() -> Optional[Dict[str, Any]]:
        response = api_client.get(
            f"/api/public/traces/{trace_id}"
        )

        if response.status_code == 404:
            return None

        if response.status_code != 200:
            raise VerificationError(
                "Trace lookup returned "
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        trace_data = response.json()

        if not isinstance(trace_data, dict):
            return None

        if (
            predicate is not None
            and not predicate(trace_data)
        ):
            return None

        return trace_data

    return wait_for(
        description or f"trace {trace_id}",
        read,
    )


def wait_for_trace_observation(
    trace_id: str,
    predicate: Callable[
        [Dict[str, Any]],
        bool,
    ],
    description: str,
) -> Dict[str, Any]:
    def read() -> Optional[Dict[str, Any]]:
        payload = request_json(
            "/api/public/observations",
            params={
                "traceId": trace_id,
                "limit": 100,
            },
        )

        observations = payload.get("data", [])

        for observation in observations:
            if (
                isinstance(observation, dict)
                and predicate(observation)
            ):
                return observation

        return None

    return wait_for(
        description,
        read,
    )


def observation_cost(
    observation: Dict[str, Any],
) -> float:
    cost_details = (
        observation.get("costDetails") or {}
    )

    candidates = [
        observation.get(
            "calculatedTotalCost"
        ),
        observation.get("totalCost"),
        cost_details.get("total"),
    ]

    for value in candidates:
        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 0.0


def observation_latency(
    observation: Dict[str, Any],
) -> Optional[float]:
    value = observation.get("latency")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def json_equivalent(
    actual: Any,
    expected_json: str,
) -> bool:
    try:
        expected = json.loads(expected_json)

        if isinstance(actual, str):
            actual = json.loads(actual)

        return actual == expected

    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


# ============================================================
# Telemetry creation helpers
# ============================================================

def create_generation_observation(
    *,
    name: str,
    model: str,
    feature: str,
    tenant: str,
    input_tokens: int,
    output_tokens: int,
    duration_seconds: float = 0,
) -> Dict[str, Any]:
    with tracer.start_as_current_span(
        name
    ) as span:
        trace_id, observation_id = span_ids(
            span
        )

        # These standard GenAI attributes cause
        # AgentGuard to classify this as GENERATION.
        span.set_attribute(
            "gen_ai.system",
            "openai",
        )

        span.set_attribute(
            "gen_ai.operation.name",
            "chat",
        )

        span.set_attribute(
            "gen_ai.request.model",
            model,
        )

        span.set_attribute(
            "gen_ai.response.model",
            model,
        )

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

        add_test_attributes(
            span,
            test_case=name,
            feature=feature,
            tenant=tenant,
        )

        if duration_seconds > 0:
            time.sleep(duration_seconds)

        span.set_status(
            Status(StatusCode.OK)
        )

    return {
        "trace_id": trace_id,
        "observation_id": observation_id,
        "name": name,
        "model": model,
        "feature": feature,
        "tenant": tenant,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration": duration_seconds,
    }


def create_basic_span(
    *,
    name: str,
    feature: str,
    tenant: str,
    duration_seconds: float,
) -> Dict[str, Any]:
    with tracer.start_as_current_span(
        name
    ) as span:
        trace_id, observation_id = span_ids(
            span
        )

        add_test_attributes(
            span,
            test_case=name,
            feature=feature,
            tenant=tenant,
        )

        span.set_attribute(
            "agentguard.operation.result",
            "success",
        )

        time.sleep(duration_seconds)

        span.set_status(
            Status(StatusCode.OK)
        )

    return {
        "trace_id": trace_id,
        "observation_id": observation_id,
        "name": name,
        "duration": duration_seconds,
    }


def create_tool_observation(
    *,
    name: str,
    tool_name: str,
    arguments: str,
    result: str,
) -> Dict[str, Any]:
    with tracer.start_as_current_span(
        name
    ) as span:
        trace_id, observation_id = span_ids(
            span
        )

        # Standard GenAI tool attributes cause
        # AgentGuard to classify this as TOOL.
        span.set_attribute(
            "gen_ai.tool.name",
            tool_name,
        )

        span.set_attribute(
            "gen_ai.tool.call.arguments",
            arguments,
        )

        span.set_attribute(
            "gen_ai.tool.call.result",
            result,
        )

        add_test_attributes(
            span,
            test_case=name,
            feature="tool-testing",
            tenant="tool-test-tenant",
        )

        time.sleep(0.1)

        span.set_status(
            Status(StatusCode.OK)
        )

    return {
        "trace_id": trace_id,
        "observation_id": observation_id,
        "name": name,
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result,
    }


def create_error_observation(
    *,
    name: str,
    message: str,
    feature: str = "error-testing",
    tenant: str = "error-test-tenant",
    tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    with tracer.start_as_current_span(
        name
    ) as span:
        trace_id, observation_id = span_ids(
            span
        )

        parent_context = span.parent
        parent_observation_id = None

        if (
            parent_context is not None
            and parent_context.is_valid
        ):
            parent_observation_id = format(
                parent_context.span_id,
                "016x",
            )

        if tool_name:
            span.set_attribute(
                "gen_ai.tool.name",
                tool_name,
            )

        add_test_attributes(
            span,
            test_case=name,
            feature=feature,
            tenant=tenant,
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
            Status(
                StatusCode.ERROR,
                message,
            )
        )

    return {
        "trace_id": trace_id,
        "observation_id": observation_id,
        "parent_observation_id": (
            parent_observation_id
        ),
        "name": name,
        "message": message,
        "expected_type": (
            "TOOL" if tool_name else "SPAN"
        ),
    }


def run_real_openai_call(
    *,
    test_case: str,
    model: str,
    feature: str,
    tenant: str,
    prompt: str,
) -> Dict[str, Any]:
    root_name = f"{test_case}-root"

    with tracer.start_as_current_span(
        root_name
    ) as root_span:
        trace_id, root_observation_id = (
            span_ids(root_span)
        )

        add_test_attributes(
            root_span,
            test_case=test_case,
            feature=feature,
            tenant=tenant,
        )

        with agentguard.policy(
            disable=["prompt-injection"],
            user_id=tenant,
            session_id=(
                f"session-{RUN_ID}-{test_case}"
            ),
            feature=feature,
            metadata={
                "tenant": tenant,
                "test_case": test_case,
                "run_id": RUN_ID,
            },
        ):
            response = (
                openai_client
                .chat
                .completions
                .create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_completion_tokens=50,
                )
            )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        root_span.set_status(
            Status(StatusCode.OK)
        )

    return {
        "trace_id": trace_id,
        "root_observation_id": (
            root_observation_id
        ),
        "test_case": test_case,
        "model": model,
        "feature": feature,
        "tenant": tenant,
        "content": content,
    }


def flush_telemetry() -> None:
    agentguard.flush()


# ============================================================
# 1. Cost by Model
# ============================================================

def test_cost_by_model() -> None:
    cases = [
        ("cost-model-1", MODEL_1),
        ("cost-model-2", MODEL_2),
        ("cost-model-3", MODEL_3),
    ]

    artifacts = []

    for test_case, model in cases:
        artifact = run_real_openai_call(
            test_case=(
                f"{test_case}-{RUN_ID}"
            ),
            model=model,
            feature="model-cost-test",
            tenant="model-test-tenant",
            prompt=(
                "Reply with exactly: "
                "MODEL TEST OK"
            ),
        )

        artifacts.append(artifact)

    flush_telemetry()

    for artifact in artifacts:
        expected_model = artifact["model"]

        observation = (
            wait_for_trace_observation(
                artifact["trace_id"],
                lambda item: (
                    item.get("type")
                    == "GENERATION"
                    and item.get("model")
                    == expected_model
                    and observation_cost(item) > 0
                    and item.get(
                        "totalTokens",
                        0,
                    ) > 0
                ),
                (
                    "positive model cost for "
                    f"{expected_model}"
                ),
            )
        )

        print(
            f"[PASS] {expected_model}: "
            f"cost="
            f"{observation_cost(observation):.8f}, "
            f"tokens="
            f"{observation.get('totalTokens')}"
        )


# ============================================================
# 2. Cost by Feature
# ============================================================

def test_cost_by_feature() -> None:
    cases = [
        (
            "classify-intent",
            500,
            50,
        ),
        (
            "summarize-document",
            1500,
            200,
        ),
        (
            "customer-support",
            800,
            120,
        ),
    ]

    artifacts = []

    for (
        feature,
        input_tokens,
        output_tokens,
    ) in cases:
        artifact = (
            create_generation_observation(
                name=feature,
                model=MODEL_1,
                feature=feature,
                tenant=(
                    "feature-test-tenant"
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

        artifacts.append(artifact)

    flush_telemetry()

    for artifact in artifacts:
        expected_total = (
            artifact["input_tokens"]
            + artifact["output_tokens"]
        )

        observation = wait_for_observation(
            artifact["observation_id"],
            predicate=lambda item: (
                item.get("type")
                == "GENERATION"
                and item.get("name")
                == artifact["name"]
                and item.get("totalTokens")
                == expected_total
                and observation_cost(item) > 0
            ),
            description=(
                "feature cost for "
                f"{artifact['feature']}"
            ),
        )

        print(
            f"[PASS] {artifact['feature']}: "
            f"cost="
            f"{observation_cost(observation):.8f}, "
            f"tokens="
            f"{observation.get('totalTokens')}"
        )


# ============================================================
# 3. Cost by Tenant
# ============================================================

def test_cost_by_tenant() -> None:
    tenants = [
        "actaclad-corp",
        "aanseaa-inc",
        "agentguard-ltd",
    ]

    artifacts = []

    for tenant in tenants:
        artifact = run_real_openai_call(
            test_case=(
                f"cost-tenant-"
                f"{tenant}-{RUN_ID}"
            ),
            model=MODEL_1,
            feature="customer-support",
            tenant=tenant,
            prompt=(
                "Reply with exactly: "
                f"{tenant.upper()} OK"
            ),
        )

        artifacts.append(artifact)

    flush_telemetry()

    for artifact in artifacts:
        tenant = artifact["tenant"]

        trace_data = wait_for_trace(
            artifact["trace_id"],
            predicate=lambda item: (
                item.get("userId") == tenant
                and float(
                    item.get("totalCost") or 0
                ) > 0
            ),
            description=(
                "positive trace cost for "
                f"tenant {tenant}"
            ),
        )

        total_cost = float(
            trace_data.get("totalCost") or 0
        )

        print(
            f"[PASS] {tenant}: "
            f"userId="
            f"{trace_data.get('userId')}, "
            f"cost={total_cost:.8f}"
        )


# ============================================================
# 4. Trace Errors
# ============================================================

def test_trace_errors() -> None:
    cases = [
        (
            "trace-error-validation",
            "Customer ID is missing",
        ),
        (
            "trace-error-timeout",
            "Payment provider timed out",
        ),
        (
            "trace-error-dependency",
            "Inventory service is unavailable",
        ),
    ]

    artifacts = []

    for name, message in cases:
        artifact = create_error_observation(
            name=f"{name}-{RUN_ID}",
            message=message,
            feature="trace-error-testing",
            tenant="trace-error-tenant",
        )

        artifacts.append(artifact)

    flush_telemetry()

    for artifact in artifacts:
        observation = wait_for_observation(
            artifact["observation_id"],
            predicate=lambda item: (
                item.get("level") == "ERROR"
                and item.get("statusMessage")
                == artifact["message"]
            ),
            description=(
                "trace error "
                f"{artifact['name']}"
            ),
        )

        trace_data = wait_for_trace(
            artifact["trace_id"]
        )

        observation_ids = {
            item.get("id")
            for item in trace_data.get(
                "observations",
                [],
            )
            if isinstance(item, dict)
        }

        assert_true(
            artifact["observation_id"]
            in observation_ids,
            (
                "Error observation "
                f"{artifact['observation_id']} "
                "is not attached to its trace"
            ),
        )

        print(
            f"[PASS] {artifact['name']}: "
            f"level="
            f"{observation.get('level')}, "
            f"message="
            f"{observation.get('statusMessage')}"
        )


# ============================================================
# 5. Span Observations
# ============================================================

def test_spans() -> None:
    cases = [
        (
            "retrieve-customer",
            "customer-lookup",
            0.10,
        ),
        (
            "build-prompt",
            "prompt-builder",
            0.20,
        ),
        (
            "parse-response",
            "response-parser",
            0.15,
        ),
    ]

    artifacts = []

    for name, feature, duration in cases:
        artifact = create_basic_span(
            name=f"{name}-{RUN_ID}",
            feature=feature,
            tenant="span-test-tenant",
            duration_seconds=duration,
        )

        artifacts.append(artifact)

    flush_telemetry()

    for artifact in artifacts:
        observation = wait_for_observation(
            artifact["observation_id"],
            predicate=lambda item: (
                item.get("type") == "SPAN"
                and observation_latency(item)
                is not None
            ),
            description=(
                "SPAN observation "
                f"{artifact['name']}"
            ),
        )

        latency = observation_latency(
            observation
        )

        expected = artifact["duration"]

        tolerance = max(
            0.10,
            expected * 0.50,
        )

        assert_true(
            abs(latency - expected)
            <= tolerance,
            (
                f"{artifact['name']} "
                f"latency {latency}s was "
                f"not close to expected "
                f"{expected}s"
            ),
        )

        print(
            f"[PASS] {artifact['name']}: "
            f"type=SPAN, "
            f"latency={latency:.3f}s"
        )


# ============================================================
# 6. Tool Observations
# ============================================================

def test_tools() -> None:
    cases = [
        (
            "weather-search",
            "weather_search",
            '{"city": "Chennai"}',
            (
                '{"temperature": 31, '
                '"unit": "celsius"}'
            ),
        ),
        (
            "order-lookup",
            "order_lookup",
            '{"order_id": "ORD-1001"}',
            '{"status": "shipped"}',
        ),
        (
            "calculator",
            "calculator",
            '{"expression": "125 * 1.18"}',
            '{"value": 147.5}',
        ),
    ]

    artifacts = []

    for (
        name,
        tool_name,
        arguments,
        result,
    ) in cases:
        artifact = create_tool_observation(
            name=f"{name}-{RUN_ID}",
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )

        artifacts.append(artifact)

    flush_telemetry()

    for artifact in artifacts:
        observation = wait_for_observation(
            artifact["observation_id"],
            predicate=lambda item: (
                item.get("type") == "TOOL"
            ),
            description=(
                "TOOL observation "
                f"{artifact['name']}"
            ),
        )

        assert_true(
            observation.get("name")
            == artifact["tool_name"],
            (
                "Expected tool name "
                f"{artifact['tool_name']}, "
                "received "
                f"{observation.get('name')}"
            ),
        )

        assert_true(
            json_equivalent(
                observation.get("input"),
                artifact["arguments"],
            ),
            (
                "Tool arguments were not "
                "stored correctly for "
                f"{artifact['tool_name']}"
            ),
        )

        assert_true(
            json_equivalent(
                observation.get("output"),
                artifact["result"],
            ),
            (
                "Tool result was not "
                "stored correctly for "
                f"{artifact['tool_name']}"
            ),
        )

        print(
            f"[PASS] "
            f"{artifact['tool_name']}: "
            "type=TOOL, "
            "input/output verified"
        )


# ============================================================
# 7. Error Observations
# ============================================================

def test_error_observations() -> None:
    parent_name = (
        "error-observation-test-suite-"
        f"{RUN_ID}"
    )

    with tracer.start_as_current_span(
        parent_name
    ) as parent_span:
        (
            parent_trace_id,
            parent_observation_id,
        ) = span_ids(parent_span)

        add_test_attributes(
            parent_span,
            test_case=parent_name,
            feature="error-testing",
            tenant="error-test-tenant",
        )

        children = [
            create_error_observation(
                name=(
                    "json-parser-error-"
                    f"{RUN_ID}"
                ),
                message=(
                    "Invalid JSON returned "
                    "by the model"
                ),
            ),
            create_error_observation(
                name=(
                    "tool-execution-error-"
                    f"{RUN_ID}"
                ),
                message=(
                    "Tool returned HTTP 500"
                ),
                tool_name="order_lookup",
            ),
            create_error_observation(
                name=(
                    "output-validation-error-"
                    f"{RUN_ID}"
                ),
                message=(
                    "Required output field "
                    "'answer' is missing"
                ),
            ),
        ]

        parent_span.set_status(
            Status(StatusCode.OK)
        )

    flush_telemetry()

    parent = wait_for_observation(
        parent_observation_id,
        predicate=lambda item: (
            item.get("type") == "SPAN"
            and item.get("level") != "ERROR"
        ),
        description=(
            "successful parent "
            "error-test span"
        ),
    )

    assert_true(
        parent.get("traceId")
        == parent_trace_id,
        (
            "Parent observation has "
            "an unexpected trace ID"
        ),
    )

    for child in children:
        observation = wait_for_observation(
            child["observation_id"],
            predicate=lambda item: (
                item.get("level") == "ERROR"
                and item.get("statusMessage")
                == child["message"]
            ),
            description=(
                "child error "
                f"{child['name']}"
            ),
        )

        assert_true(
            observation.get("traceId")
            == parent_trace_id,
            (
                f"{child['name']} is not "
                "in the parent trace"
            ),
        )

        assert_true(
            observation.get(
                "parentObservationId"
            )
            == parent_observation_id,
            (
                f"{child['name']} has "
                "the wrong parent observation"
            ),
        )

        assert_true(
            observation.get("type")
            == child["expected_type"],
            (
                f"{child['name']} expected "
                f"type "
                f"{child['expected_type']}, "
                "received "
                f"{observation.get('type')}"
            ),
        )

        print(
            f"[PASS] {child['name']}: "
            f"type="
            f"{observation.get('type')}, "
            "level=ERROR, "
            "parent verified"
        )


# ============================================================
# 8. Latency Observations
# ============================================================

def test_latency() -> None:
    cases = [
        (
            "latency-fast-200ms",
            0.20,
        ),
        (
            "latency-medium-800ms",
            0.80,
        ),
        (
            "latency-slow-1500ms",
            1.50,
        ),
    ]

    artifacts = []

    for name, duration in cases:
        artifact = (
            create_generation_observation(
                name=f"{name}-{RUN_ID}",
                model=MODEL_1,
                feature="latency-test",
                tenant=(
                    "latency-test-tenant"
                ),
                input_tokens=1,
                output_tokens=1,
                duration_seconds=duration,
            )
        )

        artifacts.append(artifact)

    flush_telemetry()

    for artifact in artifacts:
        observation = wait_for_observation(
            artifact["observation_id"],
            predicate=lambda item: (
                item.get("type")
                == "GENERATION"
                and observation_latency(item)
                is not None
            ),
            description=(
                "latency observation "
                f"{artifact['name']}"
            ),
        )

        latency = observation_latency(
            observation
        )

        expected = artifact["duration"]

        tolerance = max(
            0.15,
            expected * 0.35,
        )

        assert_true(
            abs(latency - expected)
            <= tolerance,
            (
                f"{artifact['name']} stored "
                f"latency {latency}s was not "
                f"close to expected "
                f"{expected}s"
            ),
        )

        print(
            f"[PASS] {artifact['name']}: "
            f"expected={expected:.3f}s, "
            f"stored={latency:.3f}s"
        )


# ============================================================
# Category runner
# ============================================================

def run_category(
    name: str,
    test_function: Callable[[], None],
) -> Dict[str, str]:
    print_section(name.upper())

    try:
        test_function()

        print(f"\n[PASS] {name}")

        return {
            "name": name,
            "status": "PASS",
            "message": "",
        }

    except Exception as error:
        message = (
            f"{type(error).__name__}: {error}"
        )

        print(
            f"\n[FAIL] {name}: {message}"
        )

        return {
            "name": name,
            "status": "FAIL",
            "message": message,
        }


# ============================================================
# Main
# ============================================================

def main() -> int:
    print_section(
        "AGENTGUARD OBSERVABILITY "
        "END-TO-END TEST"
    )

    print(f"Run ID      : {RUN_ID}")
    print(f"Environment : {ENVIRONMENT}")
    print(f"Base URL    : {BASE_URL}")
    print(f"Timeout     : {TIMEOUT_SECONDS}s")

    try:
        validate_agentguard_authentication()

    except Exception as error:
        print(
            f"\n[FATAL] "
            f"{type(error).__name__}: "
            f"{error}"
        )

        api_client.close()
        return 2

    categories = [
        (
            "Cost by model",
            test_cost_by_model,
        ),
        (
            "Cost by feature",
            test_cost_by_feature,
        ),
        (
            "Cost by tenant",
            test_cost_by_tenant,
        ),
        (
            "Trace errors",
            test_trace_errors,
        ),
        (
            "Span observations",
            test_spans,
        ),
        (
            "Tool observations",
            test_tools,
        ),
        (
            "Error observations",
            test_error_observations,
        ),
        (
            "Latency observations",
            test_latency,
        ),
    ]

    results: List[Dict[str, str]] = []

    try:
        for (
            category_name,
            test_function,
        ) in categories:
            result = run_category(
                category_name,
                test_function,
            )

            results.append(result)

    finally:
        try:
            flush_telemetry()
        finally:
            api_client.close()

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    failed = sum(
        result["status"] == "FAIL"
        for result in results
    )

    print_section(
        "OBSERVABILITY TEST SUMMARY"
    )

    for result in results:
        print(
            f"{result['status']:<5} "
            f"{result['name']:<25} "
            f"{result['message']}"
        )

    print("-" * 72)
    print(f"TOTAL  : {len(results)}")
    print(f"PASSED : {passed}")
    print(f"FAILED : {failed}")
    print("-" * 72)

    if failed:
        print(
            "\n[DONE] Observability "
            "verification failed."
        )

        return 1

    print(
        "\n[DONE] All observability "
        "checks passed."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())