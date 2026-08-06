"""
Unified AgentGuard + Anthropic Claude SDK test suite.

Uses:
    agentguard.init()
    agentguard.policy()
    agentguard.flush()

Does not use:
    agentguard.get_client()
    AgentGuardClient
    Langfuse client imports

Tests:
- Observability
- Prompt creation
- Prompt versioning
- Prompt datasets
- Score storage/retrieval
- LLM-as-a-Judge
- Five evaluator criteria
- Human review queue
- Datasets and experiments
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
import anthropic
from dotenv import load_dotenv

# Environment must be loaded before AgentGuard initialization.
load_dotenv(override=True)

# Allows AgentGuard to capture request/response content.
os.environ.setdefault("AGENTGUARD_CAPTURE_CONTENT", "true")

import agentguard
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


# ============================================================
# Configuration
# ============================================================

BASE_URL = os.getenv("AGENTGUARD_BASE_URL", "").rstrip("/")
PUBLIC_KEY = os.getenv("AGENTGUARD_PUBLIC_KEY")
SECRET_KEY = os.getenv("AGENTGUARD_SECRET_KEY")
PROJECT_ID = os.getenv("AGENTGUARD_PROJECT_ID")
ENVIRONMENT = os.getenv(
    "AGENTGUARD_ENVIRONMENT",
    "production",
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL = os.getenv(
    "CLAUDE_TEST_MODEL",
    "claude-sonnet-5",
)

MODEL_1 = os.getenv(
    "CLAUDE_TEST_MODEL_1",
    "claude-haiku-4-5",
)
MODEL_2 = os.getenv(
    "CLAUDE_TEST_MODEL_2",
    "claude-sonnet-5",
)
MODEL_3 = os.getenv(
    "CLAUDE_TEST_MODEL_3",
    "claude-opus-5",
)

HTTP_TIMEOUT = float(
    os.getenv("AGENTGUARD_HTTP_TIMEOUT", "30")
)

POLL_TIMEOUT = int(
    os.getenv("AGENTGUARD_POLL_TIMEOUT", "60")
)

POLL_INTERVAL = float(
    os.getenv("AGENTGUARD_POLL_INTERVAL", "2")
)

RUN_ID = (
    f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-"
    f"{uuid.uuid4().hex[:8]}"
)

SHORT_ID = uuid.uuid4().hex[:8]
SESSION_ID = f"session-{RUN_ID}"
DEFAULT_TENANT = "agentguard-combined-test"

PROMPT_CREATION_NAME = f"prompt-create-{RUN_ID}"
PROMPT_VERSION_NAME = f"prompt-version-{RUN_ID}"
PROMPT_DATASET_NAME = f"prompt-dataset-{RUN_ID}"

EVALUATION_DATASET_NAME = f"evaluation-dataset-{RUN_ID}"
EXPERIMENT_NAME = f"evaluation-run-{RUN_ID}"

TEST_RESULTS: List[Dict[str, Any]] = []


# ============================================================
# Environment validation
# ============================================================

def validate_environment() -> None:
    required = {
        "AGENTGUARD_BASE_URL": BASE_URL,
        "AGENTGUARD_PUBLIC_KEY": PUBLIC_KEY,
        "AGENTGUARD_SECRET_KEY": SECRET_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


validate_environment()


# ============================================================
# AgentGuard unified initialization
# ============================================================

agentguard.init(
    public_key=PUBLIC_KEY,
    secret_key=SECRET_KEY,
    base_url=BASE_URL,
    project_id=PROJECT_ID,
    model=MODEL,
    environment=ENVIRONMENT,
    guardrails="auto",
    streaming="chunk",
)


# ============================================================
# Clients
# ============================================================

claude_client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
)

rest_client = httpx.Client(
    base_url=BASE_URL,
    auth=httpx.BasicAuth(
        username=PUBLIC_KEY,
        password=SECRET_KEY,
    ),
    headers={
        "Accept": "application/json",
        "Content-Type": "application/json",
    },
    timeout=HTTP_TIMEOUT,
)

tracer = trace.get_tracer(
    "agentguard-unified-claude-test"
)


# ============================================================
# Test helpers
# ============================================================

class SkipTest(Exception):
    pass


def section(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_equal(
    actual: Any,
    expected: Any,
    message: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            f"{message}. Expected {expected!r}, got {actual!r}"
        )


def check_score(value: Any, name: str) -> None:
    check(
        isinstance(value, (int, float)),
        f"{name} must be numeric",
    )

    numeric_value = float(value)

    check(
        0 <= numeric_value <= 1,
        f"{name} must be between 0 and 1; "
        f"received {numeric_value}",
    )


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def policy(
    test_case: str,
    feature: str,
    *,
    tenant: str = DEFAULT_TENANT,
    model: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    """
    Create the AgentGuard policy used by this controlled test suite.
    """

    selected_model = model or MODEL

    return agentguard.policy(
        disable=[
            "prompt-injection",
            "toxic-content",
        ],
        fail="closed",
        on_block="raise",
        streaming="chunk",
        user_id=tenant,
        session_id=SESSION_ID,
        feature=feature,
        metadata={
            "tenant": tenant,
            "testCase": test_case,
            "testRunId": RUN_ID,
            "model": selected_model,
            "environment": ENVIRONMENT,
            "sdk": "anthropic",
            "apiStyle": "agentguard-init-policy",
            **(extra_metadata or {}),
        },
    )


def flush() -> None:
    agentguard.flush()


def run_test(
    group: str,
    test_id: str,
    name: str,
    function,
) -> None:
    print(f"\n[{test_id}] {name}")

    try:
        function()

    except SkipTest as error:
        status = "SKIP"
        print(f"[SKIP] {test_id}: {error}")

    except Exception as error:
        status = "FAIL"
        print(f"[FAIL] {test_id}: {name}")
        print("Error type:", type(error).__name__)
        print("Error:", error)

    else:
        status = "PASS"
        print(f"[PASS] {test_id}: {name}")

    TEST_RESULTS.append(
        {
            "group": group,
            "id": test_id,
            "name": name,
            "status": status,
        }
    )


# ============================================================
# REST helpers
# ============================================================

def rest(
    method: str,
    path: str,
    *,
    expected=(200,),
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    skip_on_404: bool = False,
) -> Any:
    if isinstance(expected, int):
        expected = (expected,)

    response = rest_client.request(
        method,
        path,
        params=params,
        json=body,
    )

    if response.status_code == 404 and skip_on_404:
        raise SkipTest(
            f"{method} {path} is unavailable on this server"
        )

    if response.status_code not in expected:
        raise AssertionError(
            f"{method} {path} returned HTTP "
            f"{response.status_code}; expected {expected}. "
            f"Response: {response.text[:1500]}"
        )

    if not response.content:
        return None

    try:
        return response.json()
    except ValueError as error:
        raise AssertionError(
            f"{method} {path} returned non-JSON content: "
            f"{response.text[:500]}"
        ) from error


def unwrap_object(response: Any) -> Dict[str, Any]:
    if not isinstance(response, dict):
        return {}

    data = response.get("data")

    if isinstance(data, dict):
        return data

    return response


def unwrap_list(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, list):
        return response

    if not isinstance(response, dict):
        return []

    data = response.get("data")

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "items",
            "scores",
            "traces",
            "datasetItems",
            "datasetRuns",
            "queues",
        ):
            value = data.get(key)

            if isinstance(value, list):
                return value

    for key in (
        "items",
        "scores",
        "traces",
        "datasetItems",
        "datasetRuns",
        "queues",
    ):
        value = response.get(key)

        if isinstance(value, list):
            return value

    return []


def response_id(response: Any) -> Optional[str]:
    data = unwrap_object(response)

    for key in (
        "id",
        "traceId",
        "observationId",
        "scoreId",
        "datasetId",
        "datasetItemId",
        "datasetRunId",
        "queueId",
    ):
        value = data.get(key)

        if value:
            return str(value)

    return None


# ============================================================
# Connection test
# ============================================================

def verify_connection() -> None:
    with policy(
        "connection-check",
        "setup",
    ):
        response = rest(
            "GET",
            "/api/public/traces",
            expected=200,
            params={
                "page": 1,
                "limit": 1,
            },
        )

    check(
        isinstance(response, dict),
        "AgentGuard returned an invalid response",
    )

    print("AgentGuard connection successful")
    print("Base URL   :", BASE_URL)
    print("Project ID :", PROJECT_ID or "derived from API key")
    print("Environment:", ENVIRONMENT)


# ============================================================
# Anthropic Claude calls
# ============================================================

def response_text(response: Any) -> str:
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
        and getattr(block, "text", None)
    ]

    return "".join(parts).strip()


def call_claude(
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: int = 150,
    model: Optional[str] = None,
    test_case: str = "claude-request",
    feature: str = "claude-request",
    tenant: str = DEFAULT_TENANT,
    policy_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    selected_model = model or MODEL

    request: Dict[str, Any] = {
        "model": selected_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    if system:
        request["system"] = system

    with policy(
        test_case,
        feature,
        tenant=tenant,
        model=selected_model,
        extra_metadata=policy_metadata,
    ):
        response = claude_client.messages.create(**request)

    content = response_text(response)

    check(content, "Claude returned an empty response")

    return content


def score_reason_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "reason": {
                "type": "string",
            },
        },
        "required": [
            "score",
            "reason",
        ],
        "additionalProperties": False,
    }


def judge_response_schema(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    if (
        "goodAnswer" in data
        and "badAnswer" in data
    ):
        return {
            "type": "object",
            "properties": {
                "good": score_reason_schema(),
                "bad": score_reason_schema(),
            },
            "required": [
                "good",
                "bad",
            ],
            "additionalProperties": False,
        }

    if "criteria" in data:
        criteria = data["criteria"]

        return {
            "type": "object",
            "properties": {
                "evaluations": {
                    "type": "object",
                    "properties": {
                        name: score_reason_schema()
                        for name in criteria
                    },
                    "required": list(criteria.keys()),
                    "additionalProperties": False,
                },
            },
            "required": [
                "evaluations",
            ],
            "additionalProperties": False,
        }

    return score_reason_schema()


def call_json_judge(
    system_prompt: str,
    data: Dict[str, Any],
    *,
    test_case: str,
    feature: str,
) -> Dict[str, Any]:
    with policy(
        test_case,
        feature,
        model=MODEL,
        extra_metadata={
            "evaluation": True,
            "judgeProvider": "anthropic",
            "judgeModel": MODEL,
        },
    ):
        response = claude_client.messages.create(
            model=MODEL,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        data,
                        ensure_ascii=False,
                    ),
                },
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": judge_response_schema(data),
                },
            },
            temperature=0,
            max_tokens=900,
        )

    content = response_text(response)

    check(content, "Judge returned an empty response")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"Judge returned invalid JSON: {content}"
        ) from error

    check(
        isinstance(result, dict),
        "Judge result must be a JSON object",
    )

    return result


# ============================================================
# Trace and observation helpers
# ============================================================

def create_trace(
    name: str,
    input_data: Any,
    output_data: Any,
    *,
    test_case: str,
    feature: str,
    tenant: str = DEFAULT_TENANT,
    level: str = "DEFAULT",
    status_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    trace_id = str(uuid.uuid4())
    observation_id = str(uuid.uuid4())

    trace_body: Dict[str, Any] = {
        "id": trace_id,
        "name": name,
        "input": input_data,
        "output": output_data,
        "environment": ENVIRONMENT,
        "userId": tenant,
        "sessionId": SESSION_ID,
        "level": level,
        "metadata": {
            "testRunId": RUN_ID,
            "testCase": test_case,
            "feature": feature,
            "tenant": tenant,
            "provider": "anthropic",
            "model": MODEL,
            "apiStyle": "agentguard-init-policy",
            **(metadata or {}),
        },
    }

    if status_message:
        trace_body["statusMessage"] = status_message

    with policy(
        test_case,
        feature,
        tenant=tenant,
        extra_metadata=metadata,
    ):
        rest(
            "POST",
            "/api/public/traces",
            expected=(200, 201),
            body=trace_body,
        )

        observation_body: Dict[str, Any] = {
            "id": observation_id,
            "traceId": trace_id,
            "name": f"{name}-observation",
            "input": input_data,
            "output": output_data,
            "level": level,
            "metadata": {
                "testRunId": RUN_ID,
                "testCase": test_case,
                "feature": feature,
                "provider": "anthropic",
                "model": MODEL,
                **(metadata or {}),
            },
        }

        if status_message:
            observation_body["statusMessage"] = status_message

        rest(
            "POST",
            "/api/public/spans",
            expected=(200, 201),
            body=observation_body,
        )

    return {
        "traceId": trace_id,
        "observationId": observation_id,
    }


def create_synthetic_span(
    name: str,
    *,
    feature: str,
    tenant: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration: float = 0,
    error_message: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_arguments: Optional[str] = None,
    tool_result: Optional[str] = None,
) -> None:
    with policy(
        name,
        feature,
        tenant=tenant,
    ):
        with tracer.start_as_current_span(name) as span:
            span.set_attribute("gen_ai.system", "anthropic")
            span.set_attribute(
                "gen_ai.provider.name",
                "anthropic",
            )
            span.set_attribute(
                "gen_ai.operation.name",
                "chat",
            )
            span.set_attribute(
                "gen_ai.request.model",
                MODEL,
            )
            span.set_attribute(
                "gen_ai.response.model",
                MODEL,
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
            span.set_attribute(
                "agentguard.test_run_id",
                RUN_ID,
            )
            span.set_attribute(
                "app.feature",
                feature,
            )
            span.set_attribute(
                "tenant.id",
                tenant,
            )

            if tool_name:
                span.set_attribute(
                    "gen_ai.tool.name",
                    tool_name,
                )

            if tool_arguments:
                span.set_attribute(
                    "gen_ai.tool.call.arguments",
                    tool_arguments,
                )

            if tool_result:
                span.set_attribute(
                    "gen_ai.tool.call.result",
                    tool_result,
                )

            if duration:
                time.sleep(duration)

            if error_message:
                error = RuntimeError(error_message)
                span.record_exception(error)
                span.set_attribute(
                    "error.type",
                    type(error).__name__,
                )
                span.set_attribute(
                    "error.message",
                    error_message,
                )
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        error_message,
                    )
                )
            else:
                span.set_status(Status(StatusCode.OK))


# ============================================================
# Score helpers
# ============================================================

def create_score(
    name: str,
    value: float,
    *,
    trace_id: str,
    observation_id: Optional[str] = None,
    comment: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    check_score(value, name)

    score_id = str(uuid.uuid4())

    body: Dict[str, Any] = {
        "id": score_id,
        "traceId": trace_id,
        "name": name,
        "value": float(value),
        "dataType": "NUMERIC",
        "source": "API",
        "comment": comment,
        "environment": ENVIRONMENT,
        "metadata": {
            "testRunId": RUN_ID,
            "provider": "anthropic",
            "model": MODEL,
            **(metadata or {}),
        },
    }

    if observation_id:
        body["observationId"] = observation_id

    with policy(
        f"score-{name}",
        "score-storage",
        extra_metadata=metadata,
    ):
        response = rest(
            "POST",
            "/api/public/scores",
            expected=(200, 201),
            body=body,
        )

    created_id = response_id(response)

    if created_id:
        check_equal(
            created_id,
            score_id,
            "Created score ID is incorrect",
        )

    return score_id


def wait_for_score(
    score_id: str,
    timeout: int = POLL_TIMEOUT,
) -> Dict[str, Any]:
    deadline = time.time() + timeout
    last_response = None

    while time.time() < deadline:
        response = rest(
            "GET",
            "/api/public/scores",
            expected=200,
            params={
                "page": 1,
                "limit": 100,
                "scoreIds": score_id,
            },
        )

        last_response = response

        for score in unwrap_list(response):
            if str(score.get("id")) == score_id:
                return score

        time.sleep(POLL_INTERVAL)

    raise AssertionError(
        f"Score {score_id} was not visible. "
        f"Last response: {last_response}"
    )


def verify_score(
    score_id: str,
    expected_name: str,
    expected_value: float,
    expected_trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    score = wait_for_score(score_id)

    check_equal(
        score.get("name"),
        expected_name,
        "Stored score name is incorrect",
    )

    actual_value = float(score.get("value"))

    check(
        abs(actual_value - float(expected_value)) < 0.0001,
        "Stored score value is incorrect",
    )

    if expected_trace_id:
        check_equal(
            score.get("traceId"),
            expected_trace_id,
            "Stored score trace is incorrect",
        )

    return score


# ============================================================
# Observability tests
# ============================================================

def obs_claude_auto_instrumentation() -> None:
    answer = call_claude(
        (
            "Explain the purpose of AI observability "
            "in one short sentence."
        ),
        max_tokens=80,
        test_case="auto-instrumentation",
        feature="observability",
    )

    print("Claude answer:", answer)


def obs_cost_by_model() -> None:
    successful_models = []

    for selected_model in [
        MODEL_1,
        MODEL_2,
        MODEL_3,
    ]:
        try:
            answer = call_claude(
                "Reply exactly: MODEL TEST OK",
                model=selected_model,
                max_tokens=30,
                test_case=f"cost-model-{selected_model}",
                feature="model-cost",
                tenant="model-test-tenant",
            )

            successful_models.append(selected_model)
            print(selected_model, ":", answer)

        except anthropic.APIError as error:
            print(
                "Model unavailable:",
                selected_model,
                error,
            )

    check(
        successful_models,
        "None of the configured Claude models were available",
    )


def obs_cost_by_feature() -> None:
    cases = [
        ("classify-intent", 500, 50),
        ("summarize-document", 1500, 200),
        ("customer-support", 800, 120),
    ]

    for feature, input_tokens, output_tokens in cases:
        create_synthetic_span(
            f"feature-{feature}",
            feature=feature,
            tenant="feature-test-tenant",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def obs_cost_by_tenant() -> None:
    for tenant in [
        "actaclad-corp",
        "aanseaa-inc",
        "agentguard-ltd",
    ]:
        answer = call_claude(
            f"Reply exactly: {tenant.upper()} OK",
            max_tokens=30,
            test_case=f"tenant-{tenant}",
            feature="customer-support",
            tenant=tenant,
        )

        print(tenant, ":", answer)


def obs_trace_errors() -> None:
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

    for name, message in cases:
        create_synthetic_span(
            name,
            feature="trace-error-testing",
            tenant="trace-error-tenant",
            error_message=message,
        )


def obs_spans() -> None:
    for name, duration in [
        ("retrieve-customer", 0.10),
        ("build-prompt", 0.20),
        ("parse-response", 0.15),
    ]:
        create_synthetic_span(
            name,
            feature="span-test",
            tenant="span-test-tenant",
            duration=duration,
        )


def obs_tools() -> None:
    cases = [
        (
            "weather-search",
            "weather_search",
            '{"city":"Chennai"}',
            '{"temperature":31}',
        ),
        (
            "order-lookup",
            "order_lookup",
            '{"order_id":"ORD-1001"}',
            '{"status":"shipped"}',
        ),
        (
            "calculator",
            "calculator",
            '{"expression":"125 * 1.18"}',
            '{"value":147.5}',
        ),
    ]

    for (
        name,
        tool_name,
        arguments,
        result,
    ) in cases:
        create_synthetic_span(
            name,
            feature="tool-test",
            tenant="tool-test-tenant",
            tool_name=tool_name,
            tool_arguments=arguments,
            tool_result=result,
        )


def obs_error_observations() -> None:
    for name, message in [
        (
            "json-parser-error",
            "Invalid JSON returned by the model",
        ),
        (
            "tool-execution-error",
            "Tool returned HTTP 500",
        ),
        (
            "output-validation-error",
            "Required field 'answer' is missing",
        ),
    ]:
        create_synthetic_span(
            name,
            feature="error-testing",
            tenant="error-test-tenant",
            error_message=message,
        )


def obs_latency() -> None:
    for name, duration in [
        ("latency-fast-200ms", 0.20),
        ("latency-medium-800ms", 0.80),
        ("latency-slow-1500ms", 1.50),
    ]:
        create_synthetic_span(
            name,
            feature="latency-test",
            tenant="latency-test-tenant",
            duration=duration,
        )


# ============================================================
# Prompt Management
# ============================================================

def create_prompt(
    name: str,
    content: str,
    *,
    labels: List[str],
    tags: List[str],
    commit_message: str,
) -> Dict[str, Any]:
    response = rest(
        "POST",
        "/api/public/v2/prompts",
        expected=(200, 201),
        body={
            "name": name,
            "type": "text",
            "prompt": content,
            "labels": labels,
            "tags": tags,
            "config": {
                "provider": "anthropic",
                "model": MODEL,
                "temperature": 0,
            },
            "commitMessage": commit_message,
        },
    )

    result = unwrap_object(response)

    check(
        result,
        f"Prompt {name} returned an empty response",
    )

    return result


def get_prompt(
    name: str,
    *,
    version: Optional[int] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "type": "text",
    }

    if version is not None:
        params["version"] = version

    if label:
        params["label"] = label

    return unwrap_object(
        rest(
            "GET",
            f"/api/public/v2/prompts/{quote(name, safe='')}",
            expected=200,
            params=params,
        )
    )


def prompt_creation() -> None:
    content = (
        "You are a helpful customer support assistant. "
        "Answer this question: {{question}}"
    )

    with policy(
        "prompt-creation",
        "prompt-management",
    ):
        created = create_prompt(
            PROMPT_CREATION_NAME,
            content,
            labels=["prompt-creation-test"],
            tags=[
                "automated-test",
                "prompt-management",
            ],
            commit_message="Created by unified API test",
        )

        version = int(created.get("version", 1))

        retrieved = get_prompt(
            PROMPT_CREATION_NAME,
            version=version,
        )

    check_equal(
        retrieved.get("prompt"),
        content,
        "Retrieved prompt content is incorrect",
    )

    print("Prompt :", PROMPT_CREATION_NAME)
    print("Version:", version)


def prompt_versioning() -> None:
    content_v1 = (
        "Classify this customer message: "
        "{{customer_message}}"
    )

    content_v2 = (
        "Classify this message as billing, technical, "
        "account, or other. Return only the category: "
        "{{customer_message}}"
    )

    with policy(
        "prompt-versioning",
        "prompt-management",
    ):
        version_1 = create_prompt(
            PROMPT_VERSION_NAME,
            content_v1,
            labels=["development"],
            tags=["automated-test", "versioning"],
            commit_message="Initial version",
        )

        version_2 = create_prompt(
            PROMPT_VERSION_NAME,
            content_v2,
            labels=["development", "production"],
            tags=["automated-test", "versioning"],
            commit_message="Improved version",
        )

        version_1_number = int(
            version_1.get("version", 1)
        )

        version_2_number = int(
            version_2.get(
                "version",
                version_1_number + 1,
            )
        )

        retrieved_v1 = get_prompt(
            PROMPT_VERSION_NAME,
            version=version_1_number,
        )

        production = get_prompt(
            PROMPT_VERSION_NAME,
            label="production",
        )

    check_equal(
        version_2_number,
        version_1_number + 1,
        "Prompt version did not increment",
    )

    check_equal(
        retrieved_v1.get("prompt"),
        content_v1,
        "Version 1 content changed",
    )

    check_equal(
        production.get("prompt"),
        content_v2,
        "Production prompt content is incorrect",
    )

    print("Version 1:", version_1_number)
    print("Version 2:", version_2_number)


def create_dataset(
    name: str,
    description: str,
) -> Dict[str, Any]:
    return unwrap_object(
        rest(
            "POST",
            "/api/public/v2/datasets",
            expected=(200, 201),
            body={
                "name": name,
                "description": description,
                "metadata": {
                    "testRunId": RUN_ID,
                    "provider": "anthropic",
                    "model": MODEL,
                },
            },
        )
    )


def create_dataset_item(
    dataset_name: str,
    input_data: Any,
    expected_output: Any,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    item_id = str(uuid.uuid4())

    result = unwrap_object(
        rest(
            "POST",
            "/api/public/dataset-items",
            expected=(200, 201),
            body={
                "id": item_id,
                "datasetName": dataset_name,
                "input": input_data,
                "expectedOutput": expected_output,
                "metadata": {
                    "testRunId": RUN_ID,
                    **(metadata or {}),
                },
                "status": "ACTIVE",
            },
        )
    )

    if not result.get("id"):
        result["id"] = item_id

    return result


def get_dataset_items(
    dataset_name: str,
) -> List[Dict[str, Any]]:
    response = rest(
        "GET",
        "/api/public/dataset-items",
        expected=200,
        params={
            "datasetName": dataset_name,
            "page": 1,
            "limit": 100,
        },
    )

    return unwrap_list(response)


def prompt_dataset() -> None:
    with policy(
        "prompt-dataset",
        "prompt-management",
    ):
        dataset = create_dataset(
            PROMPT_DATASET_NAME,
            "Prompt-management test dataset",
        )

        dataset_item = create_dataset_item(
            PROMPT_DATASET_NAME,
            {
                "question": "How can I reset my password?",
                "customerType": "premium",
            },
            {
                "answer": (
                    "Open account settings and select "
                    "Reset password."
                )
            },
            metadata={
                "promptName": PROMPT_VERSION_NAME,
            },
        )

        items = get_dataset_items(
            PROMPT_DATASET_NAME
        )

    check(
        any(
            str(item.get("id"))
            == str(dataset_item.get("id"))
            for item in items
        ),
        "Prompt dataset item was not persisted",
    )

    print("Dataset ID     :", dataset.get("id"))
    print("Dataset item ID:", dataset_item.get("id"))


# ============================================================
# Evaluation tests
# ============================================================

def score_storage() -> None:
    trace_data = create_trace(
        "score-storage-test",
        {"question": "What is 2 + 2?"},
        {"answer": "4"},
        test_case="score-storage",
        feature="score-storage",
    )

    score_name = f"answer-quality-{SHORT_ID}"
    score_value = 0.87

    score_id = create_score(
        score_name,
        score_value,
        trace_id=trace_data["traceId"],
        observation_id=trace_data["observationId"],
        comment="Unified AgentGuard score-storage test",
    )

    verify_score(
        score_id,
        score_name,
        score_value,
        trace_data["traceId"],
    )

    print("Score ID:", score_id)


def controlled_llm_judge() -> None:
    request = (
        "A customer was charged twice. "
        "What should support do?"
    )

    good_answer = (
        "Confirm the duplicate charge, refund the extra "
        "charge, and provide a settlement estimate."
    )

    bad_answer = (
        "Tell the customer duplicate charges cannot "
        "be refunded."
    )

    judgment = call_json_judge(
        (
            "Evaluate the good and bad answers. "
            "Return a JSON object exactly in this shape: "
            '{"good":{"score":0.0,"reason":"text"},'
            '"bad":{"score":0.0,"reason":"text"}}. '
            "Scores must be between 0 and 1. "
            "The correct answer must score high and the "
            "incorrect answer must score low."
        ),
        {
            "request": request,
            "goodAnswer": good_answer,
            "badAnswer": bad_answer,
        },
        test_case="controlled-judge",
        feature="llm-as-judge",
    )

    good_score = float(
        judgment["good"]["score"]
    )

    bad_score = float(
        judgment["bad"]["score"]
    )

    check_score(good_score, "Good answer")
    check_score(bad_score, "Bad answer")

    check(
        good_score >= 0.70,
        f"Good answer scored too low: {good_score}",
    )

    check(
        bad_score <= 0.30,
        f"Bad answer scored too high: {bad_score}",
    )

    check(
        good_score > bad_score,
        "Judge did not discriminate between answers",
    )

    trace_data = create_trace(
        "controlled-llm-judge",
        {"request": request},
        {
            "answer": good_answer,
            "judgment": judgment["good"],
        },
        test_case="controlled-judge",
        feature="llm-as-judge",
        metadata={
            "judgeProvider": "anthropic",
            "judgeModel": MODEL,
        },
    )

    score_id = create_score(
        "LLM Judge Correctness",
        good_score,
        trace_id=trace_data["traceId"],
        observation_id=trace_data["observationId"],
        comment=str(
            judgment["good"]["reason"]
        ),
        metadata={
            "judgeProvider": "anthropic",
            "judgeModel": MODEL,
            "badScore": bad_score,
        },
    )

    verify_score(
        score_id,
        "LLM Judge Correctness",
        good_score,
        trace_data["traceId"],
    )

    print("Good score:", good_score)
    print("Bad score :", bad_score)


def evaluator_panel() -> None:
    criteria = {
        "Correctness": (
            "Is the answer factually correct?"
        ),
        "Helpfulness": (
            "Is the answer useful and actionable?"
        ),
        "Conciseness": (
            "Is the answer appropriately concise?"
        ),
        "Hallucination": (
            "Score 1 when there are no unsupported claims "
            "and 0 for severe hallucination."
        ),
        "Context Relevance": (
            "Is the answer relevant to the supplied context?"
        ),
    }

    request = (
        "How should support handle a duplicate charge?"
    )

    context = (
        "Duplicate charges are refundable to the original "
        "payment method within 2-3 business days."
    )

    answer = (
        "Refund the duplicate charge to the original payment "
        "method and explain the 2-3 business day settlement."
    )

    result = call_json_judge(
        (
            "Evaluate all five supplied criteria from 0 to 1. "
            "For Hallucination, 1 means no hallucination. "
            "Return only a JSON object in this shape: "
            '{"evaluations":{"Criterion":'
            '{"score":0.0,"reason":"text"}}}.'
        ),
        {
            "request": request,
            "context": context,
            "answer": answer,
            "criteria": criteria,
        },
        test_case="evaluator-panel",
        feature="external-evaluator-panel",
    )

    evaluations = result.get("evaluations")

    check(
        isinstance(evaluations, dict),
        "Judge did not return evaluations",
    )

    trace_data = create_trace(
        "external-evaluator-panel",
        {
            "request": request,
            "context": context,
        },
        {
            "answer": answer,
            "evaluations": evaluations,
        },
        test_case="evaluator-panel",
        feature="external-evaluator-panel",
        metadata={
            "judgeModel": MODEL,
        },
    )

    for name in criteria:
        check(
            name in evaluations,
            f"Missing evaluator: {name}",
        )

        evaluation = evaluations[name]
        score = float(evaluation["score"])

        check_score(score, name)

        score_id = create_score(
            name,
            score,
            trace_id=trace_data["traceId"],
            observation_id=trace_data["observationId"],
            comment=str(evaluation["reason"]),
            metadata={
                "judgeProvider": "anthropic",
                "judgeModel": MODEL,
                "evaluatorType": "external",
                "criterion": name,
            },
        )

        verify_score(
            score_id,
            name,
            score,
            trace_data["traceId"],
        )

        print(f"{name:<20}: {score}")


# ============================================================
# Human Review
# ============================================================

def human_review() -> None:
    config_name = f"HR Quality {SHORT_ID}"
    queue_name = f"HR Queue {SHORT_ID}"

    with policy(
        "human-review",
        "human-review",
    ):
        score_config = unwrap_object(
            rest(
                "POST",
                "/api/public/score-configs",
                expected=(200, 201),
                body={
                    "name": config_name,
                    "dataType": "NUMERIC",
                    "minValue": 0,
                    "maxValue": 1,
                    "description": (
                        "Human-review score created by "
                        "the unified API test"
                    ),
                },
            )
        )

        score_config_id = response_id(
            score_config
        )

        check(
            score_config_id,
            "Score configuration did not return an ID",
        )

        queue = unwrap_object(
            rest(
                "POST",
                "/api/public/annotation-queues",
                expected=(200, 201),
                body={
                    "name": queue_name,
                    "description": (
                        "Unified Python human-review queue"
                    ),
                    "scoreConfigIds": [
                        score_config_id,
                    ],
                },
            )
        )

        queue_id = response_id(queue)

        check(
            queue_id,
            "Annotation queue did not return an ID",
        )

    trace_data = create_trace(
        "human-review-candidate",
        {
            "question": (
                "Can I update my billing address?"
            )
        },
        {
            "answer": (
                "Open account settings and select Billing."
            )
        },
        test_case="human-review",
        feature="human-review",
        metadata={
            "manualReviewRequired": True,
        },
    )

    with policy(
        "human-review-queue-item",
        "human-review",
    ):
        item = rest(
            "POST",
            (
                "/api/public/annotation-queues/"
                f"{queue_id}/items"
            ),
            expected=(200, 201),
            body={
                "objectId": trace_data["traceId"],
                "objectType": "TRACE",
            },
            skip_on_404=True,
        )

    print("Queue ID:", queue_id)
    print("Item ID :", response_id(item))
    print(
        "Status  : PENDING - complete the review manually "
        "in AI Quality > Human Review"
    )


# ============================================================
# Dataset experiment
# ============================================================

def create_dataset_run_item(
    *,
    dataset_item_id: str,
    trace_id: str,
    observation_id: str,
    run_name: str,
) -> Dict[str, Any]:
    return unwrap_object(
        rest(
            "POST",
            "/api/public/dataset-run-items",
            expected=(200, 201),
            body={
                "runName": run_name,
                "runDescription": (
                    "Unified AgentGuard evaluation experiment"
                ),
                "datasetItemId": dataset_item_id,
                "traceId": trace_id,
                "observationId": observation_id,
                "createdAt": utc_now(),
                "metadata": {
                    "testRunId": RUN_ID,
                    "provider": "anthropic",
                    "model": MODEL,
                },
            },
        )
    )


def wait_for_run(
    dataset_name: str,
    run_name: str,
    expected_count: int,
) -> Dict[str, Any]:
    encoded_dataset = quote(
        dataset_name,
        safe="",
    )

    encoded_run = quote(
        run_name,
        safe="",
    )

    deadline = time.time() + POLL_TIMEOUT
    last_response = None

    while time.time() < deadline:
        response = rest_client.get(
            (
                "/api/public/datasets/"
                f"{encoded_dataset}/runs/{encoded_run}"
            )
        )

        if response.status_code == 200:
            last_response = response.json()
            result = unwrap_object(last_response)

            run_items = result.get(
                "datasetRunItems",
                [],
            )

            if len(run_items) >= expected_count:
                return result

        time.sleep(POLL_INTERVAL)

    raise AssertionError(
        "Experiment run was not visible. "
        f"Last response: {last_response}"
    )


def dataset_experiment() -> None:
    with policy(
        "dataset-creation",
        "dataset-experiment",
    ):
        dataset = create_dataset(
            EVALUATION_DATASET_NAME,
            "Unified AgentGuard evaluation experiment",
        )

        cases = [
            {
                "input": {
                    "question": (
                        "What is the capital of France?"
                    ),
                    "context": (
                        "The capital of France is Paris."
                    ),
                },
                "expected": {
                    "answer": "Paris",
                },
            },
            {
                "input": {
                    "question": (
                        "Which planet is the Red Planet?"
                    ),
                    "context": (
                        "Mars is called the Red Planet."
                    ),
                },
                "expected": {
                    "answer": "Mars",
                },
            },
        ]

        items = []

        for case in cases:
            created = create_dataset_item(
                EVALUATION_DATASET_NAME,
                case["input"],
                case["expected"],
                metadata={
                    "experimentName": EXPERIMENT_NAME,
                },
            )

            items.append(
                {
                    **case,
                    "id": created["id"],
                }
            )

        stored_items = get_dataset_items(
            EVALUATION_DATASET_NAME
        )

    stored_ids = {
        str(item.get("id"))
        for item in stored_items
    }

    for item in items:
        check(
            str(item["id"]) in stored_ids,
            f"Dataset item was not persisted: {item['id']}",
        )

    run_item_ids = []
    scores = []

    for number, item in enumerate(
        items,
        start=1,
    ):
        output = call_claude(
            (
                f"Context:\n"
                f"{item['input']['context']}\n\n"
                f"Question:\n"
                f"{item['input']['question']}"
            ),
            system=(
                "Answer using only the supplied context. "
                "Be concise."
            ),
            max_tokens=100,
            test_case=f"experiment-{number}",
            feature="dataset-experiment",
        )

        trace_data = create_trace(
            f"experiment-item-{number}",
            item["input"],
            {
                "actual": output,
                "expected": item["expected"],
            },
            test_case=f"experiment-{number}",
            feature="dataset-experiment",
            metadata={
                "datasetName": EVALUATION_DATASET_NAME,
                "datasetItemId": item["id"],
                "experimentName": EXPERIMENT_NAME,
            },
        )

        with policy(
            f"experiment-run-item-{number}",
            "dataset-experiment",
        ):
            run_item = create_dataset_run_item(
                dataset_item_id=item["id"],
                trace_id=trace_data["traceId"],
                observation_id=(
                    trace_data["observationId"]
                ),
                run_name=EXPERIMENT_NAME,
            )

        run_item_id = response_id(run_item)

        check(
            run_item_id,
            "Dataset run item did not return an ID",
        )

        run_item_ids.append(run_item_id)

        expected_answer = (
            item["expected"]["answer"].lower()
        )

        accuracy = (
            1.0
            if expected_answer in output.lower()
            else 0.0
        )

        judgment = call_json_judge(
            (
                "Compare the actual and expected answers. "
                "Return only a JSON object exactly as "
                '{"score":0.0,"reason":"text"}. '
                "The score must be between 0 and 1."
            ),
            {
                "input": item["input"],
                "actual": output,
                "expected": item["expected"],
            },
            test_case=f"experiment-judge-{number}",
            feature="dataset-experiment-judge",
        )

        judge_score = float(
            judgment["score"]
        )

        check_score(
            judge_score,
            "Experiment judge",
        )

        for name, value, comment in [
            (
                "Experiment Accuracy",
                accuracy,
                "Deterministic expected-answer check",
            ),
            (
                "Experiment LLM Judge",
                judge_score,
                str(judgment["reason"]),
            ),
        ]:
            score_id = create_score(
                name,
                value,
                trace_id=trace_data["traceId"],
                observation_id=(
                    trace_data["observationId"]
                ),
                comment=comment,
                metadata={
                    "datasetName": (
                        EVALUATION_DATASET_NAME
                    ),
                    "experimentName": (
                        EXPERIMENT_NAME
                    ),
                    "datasetItemId": item["id"],
                    "datasetRunId": (
                        run_item.get("datasetRunId")
                    ),
                },
            )

            scores.append(
                (
                    score_id,
                    name,
                    value,
                    trace_data["traceId"],
                )
            )

    stored_run = wait_for_run(
        EVALUATION_DATASET_NAME,
        EXPERIMENT_NAME,
        len(items),
    )

    stored_run_items = stored_run.get(
        "datasetRunItems",
        [],
    )

    check_equal(
        len(stored_run_items),
        len(items),
        "Experiment has the wrong item count",
    )

    stored_run_ids = {
        str(item.get("id"))
        for item in stored_run_items
    }

    for run_item_id in run_item_ids:
        check(
            str(run_item_id) in stored_run_ids,
            f"Run item missing: {run_item_id}",
        )

    for (
        score_id,
        name,
        value,
        trace_id,
    ) in scores:
        verify_score(
            score_id,
            name,
            value,
            trace_id,
        )

    print("Dataset ID:", dataset.get("id"))
    print("Run ID    :", stored_run.get("id"))
    print("Run items :", len(stored_run_items))


# ============================================================
# Summary
# ============================================================

def print_summary() -> bool:
    section("UNIFIED AGENTGUARD TEST SUMMARY")

    for result in TEST_RESULTS:
        print(
            f"{result['status']:<5} "
            f"{result['group']:<15} "
            f"{result['id']:<11} "
            f"{result['name']}"
        )

    passed = sum(
        item["status"] == "PASS"
        for item in TEST_RESULTS
    )

    failed = sum(
        item["status"] == "FAIL"
        for item in TEST_RESULTS
    )

    skipped = sum(
        item["status"] == "SKIP"
        for item in TEST_RESULTS
    )

    print("-" * 76)
    print("Total  :", len(TEST_RESULTS))
    print("Passed :", passed)
    print("Failed :", failed)
    print("Skipped:", skipped)
    print("Run ID :", RUN_ID)

    print("\nUI locations:")
    print("- Traces: Observability")
    print("- Prompts: Prompt Management")
    print("- Scores: AI Quality > Scores")
    print("- Human Review: AI Quality > Human Review")
    print("- Datasets: AI Quality > Datasets")
    print(
        "- External judge results appear under Scores; "
        "they do not create native evaluator definitions."
    )

    return failed == 0


# ============================================================
# Main
# ============================================================

def main() -> int:
    section("AGENTGUARD INIT + POLICY TEST SUITE")

    print("Base URL   :", BASE_URL)
    print("Project ID :", PROJECT_ID)
    print("Model      :", MODEL)
    print("Environment:", ENVIRONMENT)
    print("Run ID     :", RUN_ID)

    run_test(
        "Setup",
        "SETUP-01",
        "AgentGuard connection",
        verify_connection,
    )

    section("OBSERVABILITY")

    observability_tests = [
        (
            "OBS-01",
            "Claude automatic instrumentation",
            obs_claude_auto_instrumentation,
        ),
        (
            "OBS-02",
            "Cost by model",
            obs_cost_by_model,
        ),
        (
            "OBS-03",
            "Cost by feature",
            obs_cost_by_feature,
        ),
        (
            "OBS-04",
            "Cost by tenant",
            obs_cost_by_tenant,
        ),
        (
            "OBS-05",
            "Trace errors",
            obs_trace_errors,
        ),
        (
            "OBS-06",
            "Span observations",
            obs_spans,
        ),
        (
            "OBS-07",
            "Tool observations",
            obs_tools,
        ),
        (
            "OBS-08",
            "Error observations",
            obs_error_observations,
        ),
        (
            "OBS-09",
            "Latency observations",
            obs_latency,
        ),
    ]

    for test_id, name, function in observability_tests:
        run_test(
            "Observability",
            test_id,
            name,
            function,
        )

    section("PROMPT MANAGEMENT")

    prompt_tests = [
        (
            "PROMPT-01",
            "Prompt creation",
            prompt_creation,
        ),
        (
            "PROMPT-02",
            "Prompt versioning",
            prompt_versioning,
        ),
        (
            "PROMPT-03",
            "Prompt dataset",
            prompt_dataset,
        ),
    ]

    for test_id, name, function in prompt_tests:
        run_test(
            "Prompts",
            test_id,
            name,
            function,
        )

    section("EVALUATION")

    evaluation_tests = [
        (
            "EVAL-01",
            "Score storage and retrieval",
            score_storage,
        ),
        (
            "EVAL-02",
            "Controlled LLM-as-a-Judge",
            controlled_llm_judge,
        ),
        (
            "EVAL-03",
            "Five evaluator criteria",
            evaluator_panel,
        ),
        (
            "EVAL-04",
            "Human review queue",
            human_review,
        ),
        (
            "DATA-01",
            "Dataset and experiment",
            dataset_experiment,
        ),
    ]

    for test_id, name, function in evaluation_tests:
        run_test(
            "Evaluation",
            test_id,
            name,
            function,
        )

    try:
        flush()
    except Exception as error:
        print("AgentGuard flush warning:", error)

    success = print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())

    finally:
        try:
            agentguard.flush()
        except Exception:
            pass

        try:
            claude_client.close()
        except Exception:
            pass

        try:
            rest_client.close()
        except Exception:
            pass
