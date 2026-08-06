import os
import sys
import time
import uuid

from dotenv import load_dotenv

# Load AgentGuard configuration before importing the SDK. Some SDK settings
# are resolved once at import time.
load_dotenv(override=True)

# Capture prompt and response content where supported.
os.environ["AGENTGUARD_CAPTURE_CONTENT"] = "true"

import agentguard
from langfuse.api.commons.types.dataset_item import DatasetItem


# The local AgentGuard API omits mediaReferences when a dataset item has no
# media. Langfuse 4.14 requires the field, so treat an omitted value as [].
DatasetItem.model_fields["media_references"].default_factory = list
DatasetItem.model_rebuild(force=True)


# ============================================================
# Configuration and AgentGuard initialization
# ============================================================

agentguard.init(
    public_key=os.getenv("AGENTGUARD_PUBLIC_KEY"),
    secret_key=os.getenv("AGENTGUARD_SECRET_KEY"),
    base_url=os.getenv("AGENTGUARD_BASE_URL"),
    project_id=os.getenv("AGENTGUARD_PROJECT_ID"),
    guardrails="auto",
)

# Prompt management uses AgentGuard's Langfuse-compatible client. Pass its
# credentials explicitly so it does not depend on import-time settings.
client = agentguard.get_client(
    public_key=os.getenv("AGENTGUARD_PUBLIC_KEY"),
    secret_key=os.getenv("AGENTGUARD_SECRET_KEY"),
    base_url=os.getenv("AGENTGUARD_BASE_URL"),
)


# ============================================================
# Unique test data
# ============================================================

RUN_ID = (
    f"{time.strftime('%Y%m%d-%H%M%S')}-"
    f"{uuid.uuid4().hex[:8]}"
)

CREATION_PROMPT_NAME = (
    f"prompt-management-test-creation-{RUN_ID}"
)

VERSION_PROMPT_NAME = (
    f"prompt-management-test-versioning-{RUN_ID}"
)

DATASET_NAME = (
    f"prompt-management-test-dataset-{RUN_ID}"
)

TEST_TENANT = "prompt-management-test-tenant"

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
            f"{message}. "
            f"Expected {expected!r}, got {actual!r}"
        )


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def create_test_policy(
    *,
    test_case,
    feature,
):
    """Create an AgentGuard policy for one test category."""

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
            "sdk": "prompt-management-python-test",
        },
    )


def run_category(
    category_name,
    test_function,
):
    """Run one category and collect failures."""

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

    return {
        "name": category_name,
        "passed": (
            len(CATEGORY_FAILURES)
            == failures_before
        ),
    }


# ============================================================
# 1. Prompt Creation
# ============================================================

def test_prompt_creation():
    print_section("1. PROMPT CREATION")

    test_case = "prompt-creation"

    prompt_content = (
        "You are a helpful customer support assistant. "
        "Answer the following customer question: "
        "{{question}}"
    )

    print(f"\n[START] {test_case}")
    print("Prompt name:", CREATION_PROMPT_NAME)

    with create_test_policy(
        test_case=test_case,
        feature="prompt-creation",
    ):
        created_prompt = client.create_prompt(
            name=CREATION_PROMPT_NAME,
            type="text",
            prompt=prompt_content,
            labels=[
                "prompt-creation-test",
            ],
            tags=[
                "automated-test",
                "prompt-management",
            ],
            config={
                "model": "gpt-4o-mini",
                "temperature": 0.2,
            },
            commit_message=(
                "Created by Prompt Management test"
            ),
        )

        assert_equal(
            created_prompt.name,
            CREATION_PROMPT_NAME,
            "Created prompt name is incorrect",
        )

        assert_equal(
            created_prompt.prompt,
            prompt_content,
            "Created prompt content is incorrect",
        )

        assert_true(
            isinstance(created_prompt.version, int),
            "Created prompt has no valid version",
        )

        created_version = created_prompt.version

        print("Created version:", created_version)

        retrieved_prompt = client.get_prompt(
            CREATION_PROMPT_NAME,
            version=created_version,
            type="text",
            cache_ttl_seconds=0,
        )

        assert_equal(
            retrieved_prompt.name,
            CREATION_PROMPT_NAME,
            "Retrieved prompt name is incorrect",
        )

        assert_equal(
            retrieved_prompt.version,
            created_version,
            "Retrieved prompt version is incorrect",
        )

        assert_equal(
            retrieved_prompt.prompt,
            prompt_content,
            "Retrieved prompt content is incorrect",
        )

    print(f"[PASS] {test_case}")
    print("Prompt successfully created and retrieved")


# ============================================================
# 2. Prompt Versioning
# ============================================================

def test_prompt_versioning():
    print_section("2. PROMPT VERSIONING")

    test_case = "prompt-versioning"

    version_one_content = (
        "Classify this customer message: "
        "{{customer_message}}"
    )

    version_two_content = (
        "Classify the following customer message as "
        "billing, technical, account, or other. "
        "Return only the category: "
        "{{customer_message}}"
    )

    print(f"\n[START] {test_case}")
    print("Prompt name:", VERSION_PROMPT_NAME)

    with create_test_policy(
        test_case=test_case,
        feature="prompt-versioning",
    ):
        print("\nCreating version 1")

        version_one = client.create_prompt(
            name=VERSION_PROMPT_NAME,
            type="text",
            prompt=version_one_content,
            labels=[
                "development",
            ],
            tags=[
                "automated-test",
                "versioning",
            ],
            config={
                "model": "gpt-4o-mini",
                "temperature": 0,
            },
            commit_message="Initial prompt version",
        )

        print(
            "Created version:",
            version_one.version,
        )

        time.sleep(0.5)

        print("\nCreating version 2")

        version_two = client.create_prompt(
            name=VERSION_PROMPT_NAME,
            type="text",
            prompt=version_two_content,
            labels=[
                "development",
                "production",
            ],
            tags=[
                "automated-test",
                "versioning",
            ],
            config={
                "model": "gpt-4o-mini",
                "temperature": 0,
            },
            commit_message=(
                "Improved classification instructions"
            ),
        )

        print(
            "Created version:",
            version_two.version,
        )

        assert_equal(
            version_two.version,
            version_one.version + 1,
            "Prompt version did not increment",
        )

        print("\nRetrieving version 1")

        retrieved_version_one = client.get_prompt(
            VERSION_PROMPT_NAME,
            version=version_one.version,
            type="text",
            cache_ttl_seconds=0,
        )

        assert_equal(
            retrieved_version_one.version,
            version_one.version,
            "Incorrect version returned for version 1",
        )

        assert_equal(
            retrieved_version_one.prompt,
            version_one_content,
            "Version 1 content was not preserved",
        )

        print("[PASS] Version 1 retrieved correctly")

        print("\nRetrieving version 2")

        retrieved_version_two = client.get_prompt(
            VERSION_PROMPT_NAME,
            version=version_two.version,
            type="text",
            cache_ttl_seconds=0,
        )

        assert_equal(
            retrieved_version_two.version,
            version_two.version,
            "Incorrect version returned for version 2",
        )

        assert_equal(
            retrieved_version_two.prompt,
            version_two_content,
            "Version 2 content is incorrect",
        )

        print("[PASS] Version 2 retrieved correctly")

        print("\nRetrieving production prompt")

        production_prompt = client.get_prompt(
            VERSION_PROMPT_NAME,
            label="production",
            type="text",
            cache_ttl_seconds=0,
        )

        assert_equal(
            production_prompt.version,
            version_two.version,
            (
                "Production label does not reference "
                "version 2"
            ),
        )

        assert_equal(
            production_prompt.prompt,
            version_two_content,
            "Production prompt content is incorrect",
        )

        print(
            "[PASS] Production label references "
            "version 2"
        )

    print(f"\n[PASS] {test_case}")


# ============================================================
# 3. Prompt Dataset
# ============================================================

def test_prompt_dataset():
    print_section("3. PROMPT DATASET")

    test_case = "prompt-dataset"

    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
            },
            "customerType": {
                "type": "string",
            },
        },
        "required": [
            "question",
            "customerType",
        ],
        "additionalProperties": False,
    }

    expected_output_schema = {
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
    }

    dataset_input = {
        "question": "How can I reset my password?",
        "customerType": "premium",
    }

    dataset_expected_output = {
        "answer": (
            "Open account settings and select "
            "'Reset password'."
        ),
    }

    print(f"\n[START] {test_case}")
    print("Dataset name:", DATASET_NAME)

    with create_test_policy(
        test_case=test_case,
        feature="prompt-dataset",
    ):
        print("\nCreating dataset")

        created_dataset = client.create_dataset(
            name=DATASET_NAME,
            description=(
                "Dataset created by the automated "
                "Prompt Management Python test"
            ),
            metadata={
                "testType": "prompt-management",
                "testRunId": RUN_ID,
                "promptName": VERSION_PROMPT_NAME,
            },
            input_schema=input_schema,
            expected_output_schema=(
                expected_output_schema
            ),
        )

        assert_equal(
            created_dataset.name,
            DATASET_NAME,
            "Created dataset name is incorrect",
        )

        assert_true(
            bool(created_dataset.id),
            "Created dataset has no ID",
        )

        dataset_id = created_dataset.id

        print("[PASS] Dataset created")
        print("Dataset ID:", dataset_id)

        print("\nCreating dataset item")

        created_item = client.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=dataset_input,
            expected_output=(
                dataset_expected_output
            ),
            metadata={
                "testCase": "password-reset",
                "testRunId": RUN_ID,
                "promptName": VERSION_PROMPT_NAME,
            },
        )

        assert_true(
            bool(created_item.id),
            "Created dataset item has no ID",
        )

        assert_equal(
            created_item.dataset_id,
            dataset_id,
            "Dataset item has incorrect dataset ID",
        )

        assert_equal(
            created_item.input,
            dataset_input,
            "Dataset item input is incorrect",
        )

        assert_equal(
            created_item.expected_output,
            dataset_expected_output,
            "Dataset expected output is incorrect",
        )

        dataset_item_id = created_item.id

        print("[PASS] Dataset item created")
        print("Dataset item ID:", dataset_item_id)

        print("\nRetrieving dataset and items")

        retrieved_dataset = client.get_dataset(
            DATASET_NAME,
            fetch_items_page_size=50,
        )

        assert_equal(
            retrieved_dataset.id,
            dataset_id,
            "Retrieved dataset ID is incorrect",
        )

        assert_equal(
            retrieved_dataset.name,
            DATASET_NAME,
            "Retrieved dataset name is incorrect",
        )

        assert_equal(
            retrieved_dataset.input_schema,
            input_schema,
            "Retrieved input schema is incorrect",
        )

        assert_equal(
            retrieved_dataset.expected_output_schema,
            expected_output_schema,
            "Retrieved output schema is incorrect",
        )

        matching_items = [
            item
            for item in retrieved_dataset.items
            if item.id == dataset_item_id
        ]

        assert_equal(
            len(matching_items),
            1,
            "Created dataset item was not found",
        )

        retrieved_item = matching_items[0]

        assert_equal(
            retrieved_item.input,
            dataset_input,
            "Retrieved dataset input is incorrect",
        )

        assert_equal(
            retrieved_item.expected_output,
            dataset_expected_output,
            "Retrieved expected output is incorrect",
        )

        assert_equal(
            retrieved_item.metadata["promptName"],
            VERSION_PROMPT_NAME,
            "Dataset prompt metadata is incorrect",
        )

    print(f"[PASS] {test_case}")


# ============================================================
# Main
# ============================================================

def main():
    print_section(
        "AGENTGUARD PROMPT MANAGEMENT TESTS STARTED"
    )

    print("Test run ID:", RUN_ID)

    categories = [
        (
            "Prompt creation",
            test_prompt_creation,
        ),
        (
            "Prompt versioning",
            test_prompt_versioning,
        ),
        (
            "Prompt dataset",
            test_prompt_dataset,
        ),
    ]

    results = []

    for category_name, test_function in categories:
        result = run_category(
            category_name,
            test_function,
        )

        results.append(result)

    print_section("FLUSHING AGENTGUARD DATA")

    try:
        agentguard.flush()
        print("[PASS] AgentGuard flush completed")

    except Exception as error:
        failure = (
            f"AgentGuard flush: "
            f"{type(error).__name__}: {error}"
        )

        CATEGORY_FAILURES.append(failure)

        print("[FAIL] AgentGuard flush")
        print("Error:", error)

    passed_count = sum(
        1
        for result in results
        if result["passed"]
    )

    failed_count = len(results) - passed_count

    print_section(
        "PROMPT MANAGEMENT TEST SUMMARY"
    )

    for result in results:
        status = (
            "PASS"
            if result["passed"]
            else "FAILED"
        )

        print(
            f"PROMPT: "
            f"{result['name']:<24}: "
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

    print("\nCreated resources:")
    print("Creation prompt:", CREATION_PROMPT_NAME)
    print("Version prompt :", VERSION_PROMPT_NAME)
    print("Dataset        :", DATASET_NAME)

    if failed_count == 0 and not CATEGORY_FAILURES:
        print(
            "\n[DONE] All Prompt Management "
            "tests passed."
        )
    else:
        print(
            "\n[DONE] One or more Prompt "
            "Management tests failed."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
