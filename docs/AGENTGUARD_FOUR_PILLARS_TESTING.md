# AgentGuard Four-Pillars Testing Guide

This document describes how this repository tests the four AgentGuard pillars: Observability, AI Quality, Security, and Governance.

## Current status

| Pillar | Status | Test implementation |
| --- | --- | --- |
| Observability | Implemented | [`Observability.py`](../Observability.py) |
| AI Quality | Implemented | [`Observability.py`](../Observability.py) |
| Security | Implemented | [`SecurityGuardrails.py`](../SecurityGuardrails.py) |
| Governance | Not started | [Governance overview](https://agentgaurd-a0acc6egbhced0dc.centralindia-01.azurewebsites.net/project/cms7u8t7n000kua07e77booma/governance-overview) |

## Shared preparation

Before running either suite:

1. Configure `.env` with AgentGuard project credentials and `OPENAI_API_KEY`.
2. Confirm the public and secret keys belong to the configured `AGENTGUARD_PROJECT_ID`.
3. Configure `AGENTGUARD_TENANT_ID`, `AGENTGUARD_TENANT_NAME`, and `AGENTGUARD_PROJECT_NAME`.
4. Ensure the AgentGuard backend and required public API routes are reachable.
5. Ensure the OpenAI account has access to the configured models and sufficient credits.

Never commit `.env` or print secret values in test output.

## Pillar 1: Observability

### Purpose

Observability proves that model activity is captured with enough context to investigate behavior, usage, performance, and failures.

### Client requirements

The application must:

- call `agentguard.init()` before OpenAI requests;
- create OpenAI calls inside `agentguard.policy()`;
- include tenant/user, session, feature, model, environment, and test metadata;
- add identity attributes to manually created spans;
- flush batched telemetry before process exit.

See [`AGENTGUARD_CLIENT_INTEGRATION.md`](AGENTGUARD_CLIENT_INTEGRATION.md) for the client code.

### Automated coverage

Run:

```powershell
python Observability.py
```

The observability group verifies:

- `OBS-01`: automatic OpenAI instrumentation;
- `OBS-02`: cost grouping by model;
- `OBS-03`: cost grouping by feature;
- `OBS-04`: cost grouping by tenant;
- `OBS-05`: error traces;
- `OBS-06`: manual span observations;
- `OBS-07`: tool observations;
- `OBS-08`: error observations;
- `OBS-09`: latency observations.

### UI verification

Open **Observability → Traces**, select a newly generated trace, and verify:

- user or tenant is not `unknown`;
- session and feature are present;
- input and output are captured according to policy;
- model and token usage are correct;
- latency, cost, status, and error information are present;
- manual spans and tool calls appear under the expected trace.

## Pillar 2: AI Quality

### Purpose

AI Quality measures whether model responses meet application expectations and makes those measurements traceable to the exact model interaction.

### Client requirements

The application should retain trace and observation IDs and use them when writing scores, evaluator results, review items, and experiment run items. Every quality record should include the tenant, dataset or test case, model, and run ID in metadata.

### Automated coverage

The same command runs AI Quality tests:

```powershell
python Observability.py
```

Coverage includes:

- prompt creation;
- prompt versioning;
- prompt datasets;
- score storage and retrieval;
- controlled LLM-as-a-Judge checks;
- five evaluation criteria;
- human-review queue creation;
- dataset and experiment execution.

### UI verification

Verify the following areas:

- **Prompt Management**: prompts, versions, labels, and datasets exist.
- **AI Quality → Scores**: scores reference the intended trace and observation.
- **AI Quality → Human Review**: queue and pending review item exist.
- **AI Quality → Datasets**: dataset, items, experiment run, and run items exist.

External judge results appear as scores. They do not automatically create native evaluator definitions in the UI.

### Backend dependencies

The deployed AgentGuard backend must expose the prompt, score, annotation-queue, dataset, dataset-run-item, and experiment retrieval routes used by `Observability.py`. A 404 skip or a polling timeout indicates a missing or incompatible backend route, not an OpenAI tracing failure.

## Pillar 3: Security

### Purpose

Security verifies that configured guardrails detect or block unsafe input and output without leaking the protected content, and that security decisions remain visible in trace and audit metadata.

### Test strategy

`SecurityGuardrails.py` loads the project guardrail configuration, then uses `disable_everything_except()` to isolate one guardrail per test. The policy uses:

```python
agentguard.policy(
    disable=disabled_guardrails,
    fail="closed",
    on_block="raise",
    streaming="chunk",
    user_id=tenant,
    session_id=session_id,
    feature=feature,
    metadata={
        "tenant": tenant,
        "projectId": project_id,
        "policyTag": f"policy:{feature}",
        "riskTag": f"risk:{risk}",
    },
)
```

Run:

```powershell
python SecurityGuardrails.py
```

### Automated coverage

The suite verifies:

- guardrail configuration and input/output scopes;
- PII detection, redaction, blocking, and false-positive protection;
- prompt-injection detection and blocking;
- toxic input and output blocking;
- secret scanner assertions and persisted block traces;
- token scanner assertions and persisted block traces;
- token-limit input and output behavior;
- JSON schema validation;
- tenant isolation;
- project credential isolation;
- policy and risk-tag persistence;
- audit activity persistence.

All secret and token samples in the suite are fake test patterns.

### Expected security results

- Benign control prompts pass.
- Malicious or prohibited test prompts raise `AgentGuardBlocked`.
- The expected guardrail name and direction appear in the stored trace.
- Raw PII, secrets, and tokens are not leaked into persisted records.
- Tenant filters do not return another tenant’s trace.
- Project credentials cannot access a different project.

If a guardrail is not enabled, the related test is skipped. If its required Input or Output scope is disabled, the configuration test fails with instructions to enable it.

## Pillar 4: Governance

### Status: not started

Governance automation has not yet been implemented in this repository.

Dashboard: [AgentGuard Governance overview](https://agentgaurd-a0acc6egbhced0dc.centralindia-01.azurewebsites.net/project/cms7u8t7n000kua07e77booma/governance-overview)

Recommended future coverage:

- model and provider inventory;
- policy ownership and approval workflow;
- risk classification and control mapping;
- exception and waiver lifecycle;
- audit-log retention and evidence export;
- project, tenant, and role-based access review;
- compliance reporting;
- change history for guardrails and policies.

Until automated tests exist, Governance should be reported as **Not started**, not `PASS` or `FAIL`.

## Release evidence checklist

For each test run, retain:

- source commit or build identifier;
- project ID and environment, without secret keys;
- run ID and tenant ID;
- terminal summary;
- links or screenshots for representative traces and scores;
- security block evidence with redacted content;
- list of skipped tests and the reason;
- Governance status and any manual review evidence.

Do not treat `SKIP` as `PASS`. Resolve or formally accept the missing dependency before release.

