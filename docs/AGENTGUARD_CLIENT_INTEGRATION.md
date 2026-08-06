# AgentGuard  Integration

This guide explains how to initialize AgentGuard, apply `agentguard.policy()`, and make OpenAI requests appear as traces in Agent Guard. The working reference implementation is [`Observability.py`](../Observability.py).

## 1. Required environment variables

Configure the client with project-scoped credentials. Do not commit real keys.

```dotenv
AGENTGUARD_BASE_URL=https://your-agentguard-host
AGENTGUARD_PROJECT_ID=actual-project-id
AGENTGUARD_PUBLIC_KEY=project-public-key
AGENTGUARD_SECRET_KEY=project-secret-key

OPENAI_API_KEY=your-openai-api-key
```

`AGENTGUARD_PROJECT_ID` must be the real project ID associated with the public and secret keys. A display name such as `test-evals` is not a substitute for the project ID.

## 2. Initialize AgentGuard

Load configuration and initialize AgentGuard before creating the OpenAI client.

```python
import os

from dotenv import load_dotenv

load_dotenv(override=True)

os.environ.setdefault(
    "AGENTGUARD_CAPTURE_CONTENT",
    "true",
)

import agentguard
import openai


agentguard.init(
    public_key=os.getenv("AGENTGUARD_PUBLIC_KEY"),
    secret_key=os.getenv("AGENTGUARD_SECRET_KEY"),
    base_url=os.getenv("AGENTGUARD_BASE_URL"),
    project_id=os.getenv("AGENTGUARD_PROJECT_ID"),
    model=os.getenv("EVALUATION_TEST_MODEL", "gpt-4o-mini"),
    environment=os.getenv("AGENTGUARD_ENVIRONMENT", "production"),
    guardrails="auto",
    on_block="raise",
    tracing="batch",
    fail="closed",
    streaming="chunk",
)

openai_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

Important settings:

- `guardrails="auto"` loads the guardrail configuration for the project.
- `on_block="raise"` raises `AgentGuardBlocked` when a guardrail blocks content.
- `fail="closed"` blocks the request if required security evaluation cannot complete.
- `tracing="batch"` batches trace export; call `agentguard.flush()` before shutdown.
- `AGENTGUARD_CAPTURE_CONTENT=true` records supported request and response content. Apply your organization’s privacy and retention rules before enabling it in production.

## 3. Apply `agentguard.policy()`

The policy context attaches identity, session, feature, and searchable metadata to calls made inside the `with` block.

```python
import os
import uuid

import agentguard


TENANT_ID = os.getenv("AGENTGUARD_TENANT_ID", "test-evals")
RUN_ID = uuid.uuid4().hex


def policy_context(test_case, feature, tenant=TENANT_ID):
    return agentguard.policy(
        disable=[],
        fail="closed",
        on_block="raise",
        streaming="chunk",
        user_id=tenant,
        session_id=f"session-{RUN_ID}",
        feature=feature,
        metadata={
            "tenant": tenant,
            "tenantId": tenant,
            "testCase": test_case,
            "testRunId": RUN_ID,
            "provider": "openai",
            "sdk": "openai",
        },
    )
```

Policy fields:

- `disable`: guardrails disabled only for this policy scope. Use `[]` when project-configured guardrails should run.
- `user_id`: the tenant or end-user identity displayed on the trace.
- `session_id`: groups related calls into one session.
- `feature`: groups cost and activity by application feature.
- `metadata`: adds searchable business, test, provider, and environment context.

The observability test currently disables `prompt-injection` and `toxic-content` in its observability policy. That is suitable when isolating trace tests, but it does not test those security controls. Security behavior is tested separately in `SecurityGuardrails.py`.

## 4. Make an OpenAI call visible in Agent Guard

Create the OpenAI request inside the policy context:

```python
def ask_openai(prompt, test_case, feature, tenant=TENANT_ID):
    with policy_context(
        test_case,
        feature,
        tenant,
    ):
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_completion_tokens=100,
        )

    return response.choices[0].message.content
```

At application shutdown, flush pending trace data:

```python
try:
    answer = ask_openai(
        "Explain AI observability in one sentence.",
        test_case="observability-smoke-test",
        feature="customer-support",
    )
    print(answer)
finally:
    agentguard.flush()
    openai_client.close()
```

After ingestion, open the project’s **Observability → Traces** page and verify:

- the trace name and OpenAI model;
- captured input and output, when content capture is permitted;
- user ID or tenant ID;
- session ID;
- feature, environment, and policy metadata;
- token usage, latency, status, and cost when available.

Only calls made after correct initialization and policy configuration are fixed. Existing traces are not retroactively changed.

## 5. Manual OpenTelemetry spans

Manual spans should explicitly set AgentGuard/Langfuse-compatible identity attributes:

```python
from opentelemetry import trace


tracer = trace.get_tracer("my-agent-service")
session_id = f"session-{RUN_ID}"

with policy_context("manual-span", "retrieval", TENANT_ID):
    with tracer.start_as_current_span("retrieve-document") as span:
        span.set_attribute("user.id", TENANT_ID)
        span.set_attribute("session.id", session_id)
 
        span.set_attribute(
            
            TENANT_ID,
        )
```

## 6. AI Quality data

AI Quality uses trace and observation IDs as the targets for scores and evaluation results. The integration pattern is:

1. Generate a traced model response.
2. retain its `traceId` and, when applicable, `observationId`;
3. create numeric or categorical scores against that trace;
4. attach dataset and experiment metadata;
5. verify the scores under **AI Quality → Scores**.

`Observability.py` contains working helpers for:

- score creation and retrieval;
- controlled LLM-as-a-Judge scoring;
- five evaluator criteria;
- prompt creation and versioning;
- prompt datasets;
- human-review queues;
- datasets and experiment runs.

Prompt management, human-review queue items, and dataset experiment retrieval require the corresponding AgentGuard public API routes to be enabled on the deployed backend.

## 7. Minimal production checklist

- Initialize AgentGuard once, before the instrumented OpenAI client is used.
- Use project-scoped credentials from the intended project.
- Wrap every model call in a policy scope.
- Always provide stable `user_id`, `session_id`, and `feature` values.
- Never put secrets or raw credentials in metadata.
- Keep security guardrails enabled in production unless a reviewed exception requires otherwise.
- Flush batched traces during graceful shutdown.
- Confirm a newly generated trace in the Agent Guard UI.
