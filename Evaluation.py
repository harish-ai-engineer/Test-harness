import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

import httpx
from dotenv import load_dotenv


# ============================================================
# Load environment before importing AgentGuard
# ============================================================

load_dotenv(override=True)

os.environ["AGENTGUARD_CAPTURE_CONTENT"] = "true"

import agentguard
import openai


# ============================================================
# Configuration
# ============================================================

agentguard.init(
    public_key=os.getenv("AGENTGUARD_PUBLIC_KEY"),
    secret_key=os.getenv("AGENTGUARD_SECRET_KEY"),
    base_url=os.getenv("AGENTGUARD_BASE_URL"),
    project_id=os.getenv("AGENTGUARD_PROJECT_ID"),
    guardrails="auto",
)

agentguard_client = agentguard.get_client()

openai_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

rest_client = httpx.Client(
    base_url=os.getenv(
        "AGENTGUARD_BASE_URL"
    ).rstrip("/"),
    auth=httpx.BasicAuth(
        username=os.getenv(
            "AGENTGUARD_PUBLIC_KEY"
        ),
        password=os.getenv(
            "AGENTGUARD_SECRET_KEY"
        ),
    ),
    headers={
        "Accept": "application/json",
        "Content-Type": "application/json",
    },
    timeout=30,
)

MODEL = os.getenv(
    "EVALUATION_TEST_MODEL",
    "gpt-4o-mini",
)

RUN_ID = (
    f"{time.strftime('%Y%m%d-%H%M%S')}-"
    f"{uuid.uuid4().hex[:8]}"
)

SHORT_ID = uuid.uuid4().hex[:8]

DATASET_NAME = f"eval-dataset-{RUN_ID}"
EXPERIMENT_NAME = f"eval-run-{RUN_ID}"
TEST_TENANT = "evaluation-test-tenant"

CATEGORY_FAILURES = []


# ============================================================
# Common helpers
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}. Expected {expected!r}, got {actual!r}"
        )


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_score(score, name):
    assert_true(
        isinstance(score, (int, float)),
        f"{name} must be numeric",
    )

    assert_true(
        0 <= score <= 1,
        f"{name} must be between 0 and 1; got {score}",
    )


def utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def create_test_policy(*, test_case, feature):
    return agentguard.policy(
        disable=[
            "prompt-injection",
        ],
        user_id=TEST_TENANT,
        session_id=f"session-{RUN_ID}",
        feature=feature,
        metadata={
            "tenant": TEST_TENANT,
            "test_case": test_case,
            "test_run_id": RUN_ID,
            "model": MODEL,
            "sdk": "evaluation-python-test",
        },
    )


def flush_all():
    agentguard_client.flush()
    agentguard.flush()


def rest_request(
    method,
    path,
    *,
    expected_status=(200,),
    params=None,
    json_body=None,
):
    if isinstance(expected_status, int):
        expected_status = (
            expected_status,
        )

    response = rest_client.request(
        method=method,
        url=path,
        params=params,
        json=json_body,
    )

    if response.status_code not in expected_status:
        raise AssertionError(
            f"{method} {path} returned "
            f"{response.status_code}; expected "
            f"{expected_status}.\n"
            f"Response: {response.text}"
        )

    if not response.content:
        return None

    return response.json()


def verify_agentguard_connection():
    print_section("AGENTGUARD API CONNECTION")

    print(
        "Base URL:",
        os.getenv("AGENTGUARD_BASE_URL"),
    )

    scores = rest_request(
        "GET",
        "/api/public/scores",
        expected_status=200,
        params={
            "page": 1,
            "limit": 1,
        },
    )

    queues = rest_request(
        "GET",
        "/api/public/annotation-queues",
        expected_status=200,
        params={
            "page": 1,
            "limit": 1,
        },
    )

    assert_true(
        "data" in scores,
        "Scores API returned an invalid response",
    )

    assert_true(
        "data" in queues,
        "Annotation queue API returned an invalid response",
    )

    print(
        "[PASS] AgentGuard API authentication verified"
    )


def call_openai(
    *,
    system_prompt,
    user_prompt,
    max_tokens=250,
):
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_completion_tokens=max_tokens,
    )

    content = response.choices[0].message.content

    if not content:
        raise AssertionError(
            "OpenAI returned an empty response"
        )

    return content.strip()


def call_json_judge(
    *,
    system_prompt,
    evaluation_data,
):
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(
                    evaluation_data,
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        response_format={
            "type": "json_object",
        },
        max_completion_tokens=300,
    )

    content = response.choices[0].message.content

    if not content:
        raise AssertionError(
            "LLM judge returned an empty response"
        )

    try:
        return json.loads(content)

    except json.JSONDecodeError as error:
        raise AssertionError(
            f"LLM judge returned invalid JSON: {content}"
        ) from error


def run_category(category_name, test_function):
    failures_before = len(
        CATEGORY_FAILURES
    )

    try:
        test_function()

    except Exception as error:
        failure = (
            f"{category_name}: "
            f"{type(error).__name__}: {error}"
        )

        CATEGORY_FAILURES.append(failure)

        print(f"\n[FAIL] {category_name}")
        print(
            "Error type:",
            type(error).__name__,
        )
        print("Error:", error)

    return {
        "name": category_name,
        "passed": (
            len(CATEGORY_FAILURES)
            == failures_before
        ),
    }


# ============================================================
# Score helpers
# ============================================================

def submit_score(
    *,
    name,
    value,
    trace_id=None,
    observation_id=None,
    dataset_run_id=None,
    config_id=None,
    queue_id=None,
    comment="",
    metadata=None,
):
    score_id = str(uuid.uuid4())

    score_body = {
        "id": score_id,
        "name": name,
        "value": value,
        "dataType": "NUMERIC",
        "comment": comment,
        "metadata": {
            **(metadata or {}),
            "testRunId": RUN_ID,
        },
        "environment": "default",
    }

    if trace_id:
        score_body["traceId"] = trace_id

    if observation_id:
        score_body["observationId"] = (
            observation_id
        )

    if dataset_run_id:
        score_body["datasetRunId"] = (
            dataset_run_id
        )

    if config_id:
        score_body["configId"] = config_id

    if queue_id:
        score_body["queueId"] = queue_id

    response = rest_request(
        "POST",
        "/api/public/scores",
        expected_status=(
            200,
            201,
        ),
        json_body=score_body,
    )

    assert_equal(
        response["id"],
        score_id,
        "Created score ID is incorrect",
    )

    return score_id


def wait_for_score(
    score_id,
    *,
    timeout_seconds=45,
):
    deadline = (
        time.time() + timeout_seconds
    )

    last_response = None

    while time.time() < deadline:
        response = rest_request(
            "GET",
            "/api/public/scores",
            expected_status=200,
            params={
                "page": 1,
                "limit": 100,
                "scoreIds": score_id,
            },
        )

        last_response = response

        matching_scores = [
            score
            for score in response["data"]
            if score["id"] == score_id
        ]

        if matching_scores:
            return matching_scores[0]

        time.sleep(2)

    raise AssertionError(
        f"Score {score_id} was not returned within "
        f"{timeout_seconds} seconds. "
        f"Last response: {last_response}"
    )


def verify_score(
    *,
    score_id,
    expected_name,
    expected_value,
    expected_trace_id=None,
):
    stored_score = wait_for_score(
        score_id
    )

    assert_equal(
        stored_score["id"],
        score_id,
        "Stored score ID is incorrect",
    )

    assert_equal(
        stored_score["name"],
        expected_name,
        "Stored score name is incorrect",
    )

    assert_true(
        abs(
            float(stored_score["value"])
            - float(expected_value)
        ) < 0.0001,
        (
            "Stored score value is incorrect. "
            f"Expected {expected_value}, "
            f"got {stored_score['value']}"
        ),
    )

    assert_equal(
        stored_score["dataType"],
        "NUMERIC",
        "Stored score type is incorrect",
    )

    if expected_trace_id:
        assert_equal(
            stored_score["traceId"],
            expected_trace_id,
            "Stored score trace ID is incorrect",
        )

    return stored_score


# ============================================================
# Evaluator functions
# ============================================================

def accuracy_evaluator(
    *,
    output,
    expected_output,
):
    expected_answer = str(
        expected_output["answer"]
    ).strip().lower()

    output_text = str(
        output or ""
    ).strip().lower()

    passed = (
        expected_answer in output_text
    )

    return {
        "name": "Accuracy",
        "value": 1.0 if passed else 0.0,
        "comment": (
            "Expected answer found"
            if passed
            else "Expected answer not found"
        ),
        "metadata": {
            "evaluator": "deterministic-accuracy",
            "expectedAnswer": expected_answer,
        },
    }


def llm_default_evaluators(
    *,
    input_data,
    output,
    expected_output,
):
    judgment = call_json_judge(
        system_prompt=(
            "Evaluate the assistant response using "
            "the provided question, context and "
            "expected answer.\n\n"
            "Return JSON with exactly these fields:\n"
            "{\n"
            '  "hallucination_score": number,\n'
            '  "relevance_score": number,\n'
            '  "reason": string\n'
            "}\n\n"
            "hallucination_score:\n"
            "- 1 means fully grounded.\n"
            "- 0 means heavily hallucinated.\n\n"
            "relevance_score:\n"
            "- 1 means fully relevant.\n"
            "- 0 means irrelevant."
        ),
        evaluation_data={
            "input": input_data,
            "output": output,
            "expectedOutput": expected_output,
        },
    )

    hallucination_score = float(
        judgment["hallucination_score"]
    )

    relevance_score = float(
        judgment["relevance_score"]
    )

    assert_score(
        hallucination_score,
        "Hallucination",
    )

    assert_score(
        relevance_score,
        "Relevance",
    )

    reason = str(
        judgment["reason"]
    )

    return [
        {
            "name": "Hallucination",
            "value": hallucination_score,
            "comment": reason,
            "metadata": {
                "judgeModel": MODEL,
            },
        },
        {
            "name": "Relevance",
            "value": relevance_score,
            "comment": reason,
            "metadata": {
                "judgeModel": MODEL,
            },
        },
    ]


# ============================================================
# 1. Default Evaluator Scores
# ============================================================

def test_default_evaluators():
    print_section(
        "1. DEFAULT EVALUATOR SCORES"
    )

    test_case = "default-evaluators"

    input_data = {
        "question": (
            "What is the capital of France?"
        ),
        "context": (
            "France is a country in Europe. "
            "Its capital is Paris."
        ),
    }

    expected_output = {
        "answer": "Paris",
    }

    with create_test_policy(
        test_case=test_case,
        feature="default-evaluators",
    ):
        with (
            agentguard_client
            .start_as_current_observation(
                name="default-evaluator-test",
                as_type="evaluator",
                input=input_data,
                metadata={
                    "testRunId": RUN_ID,
                },
            )
        ) as span:
            model_output = call_openai(
                system_prompt=(
                    "Answer using only the context."
                ),
                user_prompt=json.dumps(
                    input_data
                ),
            )

            span.update(
                output={
                    "modelOutput": model_output,
                },
            )

            trace_id = span.trace_id
            observation_id = span.id

        flush_all()

        results = [
            accuracy_evaluator(
                output=model_output,
                expected_output=expected_output,
            ),
            *llm_default_evaluators(
                input_data=input_data,
                output=model_output,
                expected_output=expected_output,
            ),
        ]

        score_records = []

        for result in results:
            score_id = submit_score(
                name=result["name"],
                value=float(result["value"]),
                trace_id=trace_id,
                observation_id=observation_id,
                comment=result["comment"],
                metadata=result["metadata"],
            )

            score_records.append(
                {
                    "id": score_id,
                    "name": result["name"],
                    "value": result["value"],
                }
            )

            print(
                f"{result['name']:<20}: "
                f"{result['value']}"
            )

        for record in score_records:
            verify_score(
                score_id=record["id"],
                expected_name=record["name"],
                expected_value=record["value"],
                expected_trace_id=trace_id,
            )

            print(
                f"[PASS] {record['name']} "
                "score persisted"
            )

    print("[PASS] Default evaluators completed")
    print("Model output:", model_output)


# ============================================================
# 2. LLM as Judge Score
# ============================================================

def test_llm_as_judge():
    print_section(
        "2. LLM AS JUDGE SCORE"
    )

    test_case = "llm-as-judge"

    question = (
        "A customer says they were charged twice. "
        "How should support respond?"
    )

    answer = (
        "Apologize, verify the two transaction IDs, "
        "open a billing investigation, and provide "
        "an expected resolution time."
    )

    rubric = {
        "correctness": (
            "Appropriate billing-support steps"
        ),
        "helpfulness": (
            "Clear next actions"
        ),
        "tone": (
            "Professional and empathetic"
        ),
    }

    with create_test_policy(
        test_case=test_case,
        feature="llm-as-judge",
    ):
        with (
            agentguard_client
            .start_as_current_observation(
                name="llm-as-judge-test",
                as_type="evaluator",
                input={
                    "question": question,
                    "answer": answer,
                    "rubric": rubric,
                },
                metadata={
                    "judgeModel": MODEL,
                    "testRunId": RUN_ID,
                },
            )
        ) as span:
            judgment = call_json_judge(
                system_prompt=(
                    "Evaluate the answer using the rubric.\n\n"
                    "Return JSON with exactly:\n"
                    "{\n"
                    '  "score": number,\n'
                    '  "passed": boolean,\n'
                    '  "reason": string\n'
                    "}\n\n"
                    "Score must be between 0 and 1. "
                    "passed must be true only when "
                    "score is at least 0.70."
                ),
                evaluation_data={
                    "question": question,
                    "answer": answer,
                    "rubric": rubric,
                },
            )

            judge_score = float(
                judgment["score"]
            )

            judge_passed = bool(
                judgment["passed"]
            )

            judge_reason = str(
                judgment["reason"]
            )

            assert_score(
                judge_score,
                "LLM judge",
            )

            assert_equal(
                judge_passed,
                judge_score >= 0.70,
                "Judge passed value is incorrect",
            )

            span.update(
                output=judgment,
            )

            trace_id = span.trace_id
            observation_id = span.id

        flush_all()

        score_id = submit_score(
            name="LLM-as-a-Judge",
            value=judge_score,
            trace_id=trace_id,
            observation_id=observation_id,
            comment=judge_reason,
            metadata={
                "judgeModel": MODEL,
                "passed": judge_passed,
            },
        )

        verify_score(
            score_id=score_id,
            expected_name="LLM-as-a-Judge",
            expected_value=judge_score,
            expected_trace_id=trace_id,
        )

    print("[PASS] LLM judge score persisted")
    print("Score :", judge_score)
    print("Passed:", judge_passed)
    print("Reason:", judge_reason)


# ============================================================
# 3. Human Review Queue
# ============================================================

def get_or_create_review_queue():
    queues = rest_request(
        "GET",
        "/api/public/annotation-queues",
        expected_status=200,
        params={
            "page": 1,
            "limit": 50,
        },
    )

    existing_test_queues = [
        queue
        for queue in queues["data"]
        if queue["name"].startswith(
            "HR Queue "
        )
    ]

    if existing_test_queues:
        queue = existing_test_queues[0]

        print(
            "Using existing queue:",
            queue["name"],
        )

        return queue

    config_name = f"HR Quality {SHORT_ID}"
    queue_name = f"HR Queue {SHORT_ID}"

    assert_true(
        len(config_name) <= 35,
        "Score-config name is too long",
    )

    assert_true(
        len(queue_name) <= 35,
        "Queue name is too long",
    )

    score_config = rest_request(
        "POST",
        "/api/public/score-configs",
        expected_status=(
            200,
            201,
        ),
        json_body={
            "name": config_name,
            "dataType": "NUMERIC",
            "minValue": 0,
            "maxValue": 1,
            "description": (
                "Python human-review score"
            ),
        },
    )

    queue = rest_request(
        "POST",
        "/api/public/annotation-queues",
        expected_status=(
            200,
            201,
        ),
        json_body={
            "name": queue_name,
            "description": (
                "Python human-review queue"
            ),
            "scoreConfigIds": [
                score_config["id"],
            ],
        },
    )

    stored_queue = rest_request(
        "GET",
        (
            "/api/public/annotation-queues/"
            f"{queue['id']}"
        ),
        expected_status=200,
    )

    assert_equal(
        stored_queue["id"],
        queue["id"],
        "Queue was not persisted",
    )

    return stored_queue


def test_human_review_queue():
    print_section(
        "3. HUMAN REVIEW QUEUE"
    )

    test_case = "human-review"

    review_input = {
        "question": (
            "Can a customer update their "
            "billing address?"
        ),
    }

    review_output = {
        "answer": (
            "Yes. Open account settings, "
            "select Billing, and update the address."
        ),
    }

    with create_test_policy(
        test_case=test_case,
        feature="human-review",
    ):
        with (
            agentguard_client
            .start_as_current_observation(
                name="human-review-candidate",
                as_type="span",
                input=review_input,
                output=review_output,
                metadata={
                    "requiresHumanReview": True,
                    "testRunId": RUN_ID,
                },
            )
        ) as span:
            trace_id = span.trace_id
            observation_id = span.id

        flush_all()

        queue = get_or_create_review_queue()

        queue_item = rest_request(
            "POST",
            (
                "/api/public/annotation-queues/"
                f"{queue['id']}/items"
            ),
            expected_status=(
                200,
                201,
            ),
            json_body={
                "objectId": trace_id,
                "objectType": "TRACE",
                "status": "PENDING",
            },
        )

        pending_item = rest_request(
            "GET",
            (
                "/api/public/annotation-queues/"
                f"{queue['id']}/items/"
                f"{queue_item['id']}"
            ),
            expected_status=200,
        )

        assert_equal(
            pending_item["status"],
            "PENDING",
            "Queue item is not PENDING",
        )

        score_config_id = (
            queue["scoreConfigIds"][0]
        )

        human_score_id = submit_score(
            name="Human Review Quality",
            value=1.0,
            trace_id=trace_id,
            observation_id=observation_id,
            config_id=score_config_id,
            queue_id=queue["id"],
            comment=(
                "Human reviewer approved the response"
            ),
            metadata={
                "reviewStatus": "approved",
                "queueItemId": queue_item["id"],
            },
        )

        verify_score(
            score_id=human_score_id,
            expected_name="Human Review Quality",
            expected_value=1.0,
            expected_trace_id=trace_id,
        )

        completed_item = rest_request(
            "PATCH",
            (
                "/api/public/annotation-queues/"
                f"{queue['id']}/items/"
                f"{queue_item['id']}"
            ),
            expected_status=200,
            json_body={
                "status": "COMPLETED",
            },
        )

        assert_equal(
            completed_item["status"],
            "COMPLETED",
            "Queue item was not completed",
        )

        assert_true(
            completed_item["completedAt"]
            is not None,
            "Completed item has no completion time",
        )

    print("[PASS] Human review queue persisted")
    print("Queue:", queue["name"])
    print("Item :", queue_item["id"])
    print("Status: COMPLETED")


# ============================================================
# 4. Dataset and Experiment
# ============================================================

def experiment_task(input_data):
    return call_openai(
        system_prompt=(
            "Answer using only the supplied context. "
            "Keep the answer concise."
        ),
        user_prompt=(
            f"Context:\n{input_data['context']}\n\n"
            f"Question:\n{input_data['question']}"
        ),
    )


def wait_for_dataset_run(
    dataset_name,
    run_name,
    expected_items,
    *,
    timeout_seconds=45,
):
    encoded_dataset = quote(
        dataset_name,
        safe="",
    )

    encoded_run = quote(
        run_name,
        safe="",
    )

    deadline = (
        time.time() + timeout_seconds
    )

    last_response = None

    while time.time() < deadline:
        try:
            response = rest_request(
                "GET",
                (
                    "/api/public/datasets/"
                    f"{encoded_dataset}/runs/"
                    f"{encoded_run}"
                ),
                expected_status=200,
            )

            last_response = response

            if (
                len(
                    response[
                        "datasetRunItems"
                    ]
                )
                >= expected_items
            ):
                return response

        except Exception as error:
            last_response = str(error)

        time.sleep(2)

    raise AssertionError(
        "Dataset experiment run was not available "
        f"within {timeout_seconds} seconds. "
        f"Last response: {last_response}"
    )


def test_dataset_and_experiment():
    print_section(
        "4. DATASET AND EXPERIMENT"
    )

    test_case = "dataset-experiment"

    cases = [
        {
            "input": {
                "question": (
                    "What is the capital of France?"
                ),
                "context": (
                    "France is a European country. "
                    "Its capital is Paris."
                ),
            },
            "expectedOutput": {
                "answer": "Paris",
            },
            "metadata": {
                "category": "geography",
            },
        },
        {
            "input": {
                "question": (
                    "Which planet is called "
                    "the Red Planet?"
                ),
                "context": (
                    "Mars appears red because of "
                    "iron minerals in its soil."
                ),
            },
            "expectedOutput": {
                "answer": "Mars",
            },
            "metadata": {
                "category": "science",
            },
        },
    ]

    with create_test_policy(
        test_case=test_case,
        feature="dataset-experiment",
    ):
        dataset = rest_request(
            "POST",
            "/api/public/v2/datasets",
            expected_status=(
                200,
                201,
            ),
            json_body={
                "name": DATASET_NAME,
                "description": (
                    "Python evaluation dataset"
                ),
                "metadata": {
                    "testRunId": RUN_ID,
                },
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "context": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "question",
                        "context",
                    ],
                    "additionalProperties": False,
                },
                "expectedOutputSchema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "answer",
                    ],
                    "additionalProperties": False,
                },
            },
        )

        print("[PASS] Dataset created")
        print("Dataset ID:", dataset["id"])

        dataset_items = []

        for number, case in enumerate(
            cases,
            start=1,
        ):
            item = rest_request(
                "POST",
                "/api/public/dataset-items",
                expected_status=(
                    200,
                    201,
                ),
                json_body={
                    "datasetName": DATASET_NAME,
                    "input": case["input"],
                    "expectedOutput": (
                        case["expectedOutput"]
                    ),
                    "metadata": {
                        **case["metadata"],
                        "testRunId": RUN_ID,
                        "caseNumber": number,
                    },
                    "status": "ACTIVE",
                },
            )

            dataset_items.append(
                {
                    **case,
                    "id": item["id"],
                }
            )

            print(
                f"[PASS] Dataset item {number} "
                f"created: {item['id']}"
            )

        items_response = rest_request(
            "GET",
            "/api/public/dataset-items",
            expected_status=200,
            params={
                "datasetName": DATASET_NAME,
                "page": 1,
                "limit": 50,
            },
        )

        stored_item_ids = {
            item["id"]
            for item in items_response["data"]
        }

        for item in dataset_items:
            assert_true(
                item["id"] in stored_item_ids,
                (
                    "Dataset item was not persisted: "
                    f"{item['id']}"
                ),
            )

        print(
            "[PASS] Dataset items persisted"
        )

        run_item_ids = []
        experiment_score_ids = []

        for number, item in enumerate(
            dataset_items,
            start=1,
        ):
            with (
                agentguard_client
                .start_as_current_observation(
                    name=(
                        f"experiment-item-{number}"
                    ),
                    as_type="span",
                    input=item["input"],
                    metadata={
                        "datasetName": DATASET_NAME,
                        "experimentName": (
                            EXPERIMENT_NAME
                        ),
                        "datasetItemId": item["id"],
                        "testRunId": RUN_ID,
                    },
                )
            ) as span:
                output = experiment_task(
                    item["input"]
                )

                span.update(
                    output={
                        "answer": output,
                    },
                )

                trace_id = span.trace_id
                observation_id = span.id

            flush_all()

            run_item = rest_request(
                "POST",
                "/api/public/dataset-run-items",
                expected_status=(
                    200,
                    201,
                ),
                json_body={
                    "runName": EXPERIMENT_NAME,
                    "runDescription": (
                        "Python evaluation experiment"
                    ),
                    "metadata": {
                        "testRunId": RUN_ID,
                        "model": MODEL,
                    },
                    "datasetItemId": item["id"],
                    "traceId": trace_id,
                    "observationId": observation_id,
                    "createdAt": utc_now(),
                },
            )

            run_item_ids.append(
                run_item["id"]
            )

            accuracy_result = (
                accuracy_evaluator(
                    output=output,
                    expected_output=(
                        item["expectedOutput"]
                    ),
                )
            )

            judge_result = call_json_judge(
                system_prompt=(
                    "Evaluate the experiment output.\n"
                    "Return JSON with exactly:\n"
                    "{\n"
                    '  "score": number,\n'
                    '  "reason": string\n'
                    "}\n"
                    "Score must be between 0 and 1."
                ),
                evaluation_data={
                    "input": item["input"],
                    "output": output,
                    "expectedOutput": (
                        item["expectedOutput"]
                    ),
                },
            )

            judge_score = float(
                judge_result["score"]
            )

            assert_score(
                judge_score,
                "Experiment LLM Judge",
            )

            score_results = [
                accuracy_result,
                {
                    "name": (
                        "Experiment LLM Judge"
                    ),
                    "value": judge_score,
                    "comment": str(
                        judge_result["reason"]
                    ),
                },
            ]

            for score_result in score_results:
                score_id = submit_score(
                    name=score_result["name"],
                    value=float(
                        score_result["value"]
                    ),
                    trace_id=trace_id,
                    observation_id=observation_id,
                    dataset_run_id=(
                        run_item["datasetRunId"]
                    ),
                    comment=score_result[
                        "comment"
                    ],
                    metadata={
                        "datasetName": DATASET_NAME,
                        "experimentName": (
                            EXPERIMENT_NAME
                        ),
                        "datasetItemId": item["id"],
                    },
                )

                experiment_score_ids.append(
                    {
                        "id": score_id,
                        "name": score_result["name"],
                        "value": score_result["value"],
                        "traceId": trace_id,
                    }
                )

            print(
                f"[PASS] Experiment item {number} "
                "processed"
            )
            print("Output:", output)

        stored_run = wait_for_dataset_run(
            DATASET_NAME,
            EXPERIMENT_NAME,
            len(dataset_items),
        )

        assert_equal(
            len(
                stored_run[
                    "datasetRunItems"
                ]
            ),
            len(dataset_items),
            (
                "Stored experiment does not contain "
                "all run items"
            ),
        )

        for score in experiment_score_ids:
            verify_score(
                score_id=score["id"],
                expected_name=score["name"],
                expected_value=score["value"],
                expected_trace_id=(
                    score["traceId"]
                ),
            )

        print(
            "[PASS] Experiment scores persisted"
        )

    print("[PASS] Dataset experiment persisted")
    print(
        "Dataset run ID:",
        stored_run["id"],
    )
    print(
        "Run items:",
        len(
            stored_run["datasetRunItems"]
        ),
    )


# ============================================================
# 5. Score Pipeline Test
# ============================================================

def test_score_pipeline():
    print_section(
        "5. SCORE CREATION AND RETRIEVAL"
    )

    test_case = "score-pipeline"
    score_name = f"Score Test {SHORT_ID}"
    expected_value = 0.87

    with create_test_policy(
        test_case=test_case,
        feature="score-verification",
    ):
        with (
            agentguard_client
            .start_as_current_observation(
                name="score-pipeline-test",
                as_type="evaluator",
                input={
                    "test": "Score persistence",
                },
                output={
                    "expectedScore": expected_value,
                },
                metadata={
                    "testRunId": RUN_ID,
                },
            )
        ) as span:
            trace_id = span.trace_id
            observation_id = span.id

        flush_all()

        score_id = submit_score(
            name=score_name,
            value=expected_value,
            trace_id=trace_id,
            observation_id=observation_id,
            comment=(
                "Python score-pipeline test"
            ),
            metadata={
                "scorePipelineTest": True,
            },
        )

        stored_score = verify_score(
            score_id=score_id,
            expected_name=score_name,
            expected_value=expected_value,
            expected_trace_id=trace_id,
        )

        assert_equal(
            stored_score["metadata"].get(
                "scorePipelineTest"
            ),
            True,
            "Score metadata is incorrect",
        )

    print("[PASS] Score persisted and retrieved")
    print("Score ID:", score_id)
    print("Value   :", stored_score["value"])


# ============================================================
# Main
# ============================================================

def main():
    print_section(
        "AGENTGUARD EVALUATION TESTS STARTED"
    )

    print("Model      :", MODEL)
    print("Test run ID:", RUN_ID)

    try:
        verify_agentguard_connection()

    except Exception as error:
        print("\n[FAIL] AgentGuard connection")
        print(
            "Error type:",
            type(error).__name__,
        )
        print("Error:", error)
        sys.exit(1)

    categories = [
        (
            "Default evaluator scores",
            test_default_evaluators,
        ),
        (
            "LLM as Judge score",
            test_llm_as_judge,
        ),
        (
            "Human review queue",
            test_human_review_queue,
        ),
        (
            "Dataset and experiment",
            test_dataset_and_experiment,
        ),
        (
            "Score creation and retrieval",
            test_score_pipeline,
        ),
    ]

    results = []

    try:
        for category_name, test_function in categories:
            results.append(
                run_category(
                    category_name,
                    test_function,
                )
            )

    finally:
        try:
            flush_all()

        finally:
            rest_client.close()

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_count = (
        len(results) - passed_count
    )

    print_section(
        "EVALUATION TEST SUMMARY"
    )

    for result in results:
        status = (
            "PASS"
            if result["passed"]
            else "FAILED"
        )

        print(
            f"EVAL: "
            f"{result['name']:<32}: "
            f"{status}"
        )

    print("-" * 70)
    print(f"TOTAL  : {len(results)}")
    print(f"PASS   : {passed_count}")
    print(f"FAILED : {failed_count}")
    print("=" * 70)

    if CATEGORY_FAILURES:
        print("\nFAILURE DETAILS")

        for number, failure in enumerate(
            CATEGORY_FAILURES,
            start=1,
        ):
            print(f"{number}. {failure}")

    print("\nExpected UI locations:")
    print(
        "- Scores: AI Quality > Scores"
    )
    print(
        "- Queue: AI Quality > Human Review"
    )
    print(
        "- Dataset/run: AI Quality > Datasets"
    )
    print(
        "- Evaluator definitions must be "
        "created through the LLM-as-a-Judge UI"
    )

    if failed_count == 0:
        print(
            "\n[DONE] All persisted evaluation "
            "tests passed."
        )

    else:
        print(
            "\n[DONE] One or more evaluation "
            "tests failed."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()