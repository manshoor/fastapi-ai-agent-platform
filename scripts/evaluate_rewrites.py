"""Deterministic rewrite rubric and explicitly authorized network evaluation.

Default: score bundled synthetic reference suggestions OFFLINE (a rubric sanity
check, not model-quality evidence). --responses JSONL scores saved candidates with
{id,title,description}; IDs must exactly match the dataset. --provider calls the
real Responses adapter once per case/model, sequentially, without hidden retries.
--model can repeat for an actual provider comparison (all models use identical
snapshots, prompt, schema, rubric). No judge model is used. The rubric checks app
schema, required/forbidden phrases, numeric preservation, and case-specific bounds;
it cannot establish semantic correctness or detect arbitrary hallucinations.

Network gates are independent: --provider requires --allow-paid-requests plus an
explicit OPENAI_API_KEY; --publish-langsmith requires --allow-external-publication
plus LANGSMITH_API_KEY. Publication sends this SYNTHETIC dataset and candidate
outputs to the real LangSmith datasets/runs/feedback APIs, linked by example ID.
Never use private production content. Publication is not hosted UI validation.

Examples (from root, locked development environment):
  uv run --locked python scripts/evaluate_rewrites.py
  uv run --locked python scripts/evaluate_rewrites.py --responses candidates.jsonl
  uv run --locked python scripts/evaluate_rewrites.py --provider --allow-paid-requests \
    --model gpt-5.6-terra --model OTHER_AUTHORIZED_MODEL
  uv run --locked python scripts/evaluate_rewrites.py --publish-langsmith \
    --allow-external-publication --dataset-name item-rewrite-v1-synthetic

Exit 0 when every score passes, 1 for rubric/provider failures, 2 for invalid
configuration/input or publication errors. Only case IDs, models, fixed error codes,
and scores are printed, never prompts, generated content, credentials, or exception
messages. Credentials/endpoints are process environment only (no dotenv).
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import ValidationError

from app.ai.config import ProviderSettings
from app.ai.contracts import RewriteRequest, RewriteSuggestion
from app.ai.provider import OpenAIRewriter, ProviderFailure

DATASET = Path(__file__).resolve().parents[1] / "tests/evals/item_rewrite_v1.jsonl"
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)?")


def load_dataset(path=DATASET):
    cases = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if not isinstance(case, dict) or not re.fullmatch(r"[a-z0-9-]{1,64}", case["id"]):
            raise ValueError("Invalid case ID")
        if case["id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["id"])
        source = case["input"]
        RewriteSuggestion.model_validate(
            {"title": source["title"], "description": source["description"]}
        )
        RewriteRequest.model_validate({"instruction": source["instruction"]})
        RewriteSuggestion.model_validate(case["expected"])
        for key in ("required", "forbidden"):
            if not isinstance(case[key], list) or not all(isinstance(v, str) for v in case[key]):
                raise ValueError("Invalid rubric phrases")
        cases.append(case)
    if not cases:
        raise ValueError("Empty dataset")
    return cases


def score_suggestion(case, output):
    try:
        suggestion = RewriteSuggestion.model_validate(output)
    except ValidationError:
        return {
            "schema": False,
            "required_facts": False,
            "forbidden_claims": False,
            "numbers": False,
            "bounds": False,
        }
    text = f"{suggestion.title}\n{suggestion.description}".casefold()
    source = f"{case['input']['title']}\n{case['input']['description']}"
    # Preserve numeric facts exactly and reject invented numeric facts. Ordering may change.
    source_numbers = sorted(_NUMBERS.findall(source))
    output_numbers = sorted(_NUMBERS.findall(text))
    return {
        "schema": True,
        "required_facts": all(term.casefold() in text for term in case["required"]),
        "forbidden_claims": all(term.casefold() not in text for term in case["forbidden"]),
        "numbers": source_numbers == output_numbers,
        "bounds": len(suggestion.title) < case.get("title_max", 201)
        and (case["id"] != "empty-description" or suggestion.description == ""),
    }


def load_responses(path, cases):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    result = {}
    for row in rows:
        case_id = row.pop("id")
        if case_id in result:
            raise ValueError("Duplicate response ID")
        result[case_id] = row
    if set(result) != {case["id"] for case in cases}:
        raise ValueError("Response IDs must exactly match dataset")
    return result


async def evaluate(cases, *, settings=None, models=(), responses=None):
    records = []
    shared = OpenAIRewriter(settings) if settings else None
    try:
        for model in models or ("offline-reference" if responses is None else "offline-candidate",):
            provider = (
                OpenAIRewriter(
                    settings.model_copy(update={"ai_model": model}), client=shared.client
                )
                if shared
                else None
            )
            for case in cases:
                error = None
                started = datetime.now(UTC)
                try:
                    if provider:
                        generated = await provider.rewrite(**case["input"])
                        output = {"title": generated.title, "description": generated.description}
                    else:
                        output = (
                            responses[case["id"]] if responses is not None else case["expected"]
                        )
                    scores = score_suggestion(case, output)
                except ProviderFailure as exc:
                    output = {}
                    error = exc.code
                    scores = dict.fromkeys(
                        ("schema", "required_facts", "forbidden_claims", "numbers", "bounds"), False
                    )
                records.append(
                    {
                        "case_id": case["id"],
                        "model": model,
                        "scores": scores,
                        "output": output,
                        "error_code": error,
                        "started": started,
                        "finished": datetime.now(UTC),
                    }
                )
    finally:
        if shared:
            await shared.aclose()
    return records


def publish_langsmith(cases, records, *, dataset_name):
    # Importing/constructing this network client happens only after the publication gate.
    from langsmith import Client
    from langsmith.utils import LangSmithNotFoundError

    client = Client(
        api_url=os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        api_key=os.environ["LANGSMITH_API_KEY"],
        auto_batch_tracing=False,
        tracing_mode="langsmith",
        tracing_sampling_rate=1,
        hide_inputs=False,
        hide_outputs=False,
        hide_metadata=False,
        omit_traced_runtime_info=True,
        timeout_ms=(3000, 10000),
    )
    try:
        try:
            dataset = client.read_dataset(dataset_name=dataset_name)
        except LangSmithNotFoundError:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description=(
                    "Synthetic item rewrite v1; deterministic heuristic rubric, no private data."
                ),
            )
        existing = {
            str(example.id): example for example in client.list_examples(dataset_id=dataset.id)
        }
        example_ids = {}
        missing = []
        for case in cases:
            example_id = uuid5(NAMESPACE_URL, f"item-rewrite:{dataset.id}:{case['id']}")
            example_ids[case["id"]] = example_id
            prior = existing.get(str(example_id))
            if prior:
                if prior.inputs != case["input"] or prior.outputs != case["expected"]:
                    raise ValueError("Hosted dataset differs from the versioned synthetic corpus")
            else:
                missing.append(
                    {
                        "id": example_id,
                        "inputs": case["input"],
                        "reference_outputs": case["expected"],
                        "metadata": {
                            "case_id": case["id"],
                            "category": case["category"],
                            "prompt_version": "item-rewrite-v1",
                            "schema_version": "1",
                        },
                    }
                )
        if missing:
            client.create_examples(dataset_id=dataset.id, examples=missing)
        project = client.create_project(
            project_name=f"item-rewrite-eval-{uuid4().hex}",
            reference_dataset_id=dataset.id,
            metadata={"rubric": "deterministic-v1", "synthetic": True},
        )
        by_id = {case["id"]: case for case in cases}
        for record in records:
            run_id = uuid4()
            client.create_run(
                id=run_id,
                name="item-rewrite",
                run_type="chain",
                project_name=project.name,
                inputs=by_id[record["case_id"]]["input"],
                outputs=record["output"],
                start_time=record["started"],
                end_time=record["finished"],
                reference_example_id=example_ids[record["case_id"]],
                error=record["error_code"],
                extra={
                    "metadata": {
                        "model": record["model"],
                        "prompt_version": "item-rewrite-v1",
                        "schema_version": "1",
                        "rubric": "deterministic-v1",
                    }
                },
            )
            for key, passed in record["scores"].items():
                client.create_feedback(
                    run_id=run_id, key=key, score=int(passed), session_id=project.id
                )
    finally:
        client.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--provider", action="store_true")
    parser.add_argument("--allow-paid-requests", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--publish-langsmith", action="store_true")
    parser.add_argument("--allow-external-publication", action="store_true")
    parser.add_argument("--dataset-name", default="item-rewrite-v1-synthetic")
    args = parser.parse_args(argv)
    if args.provider and (not args.allow_paid_requests or not os.environ.get("OPENAI_API_KEY")):
        parser.error("Provider mode requires --allow-paid-requests and OPENAI_API_KEY")
    if args.responses and args.provider:
        parser.error("--responses and --provider are mutually exclusive")
    if args.model and not args.provider:
        parser.error("--model requires --provider")
    if args.publish_langsmith and (
        not args.allow_external_publication or not os.environ.get("LANGSMITH_API_KEY")
    ):
        parser.error("Publication requires --allow-external-publication and LANGSMITH_API_KEY")
    if args.publish_langsmith and args.dataset.resolve() != DATASET.resolve():
        parser.error("Only the bundled synthetic dataset may be published")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        cases = load_dataset(args.dataset)
        settings = ProviderSettings() if args.provider else None
        models = args.model or ([settings.ai_model] if settings else [])
        responses = load_responses(args.responses, cases) if args.responses else None
        records = asyncio.run(
            evaluate(cases, settings=settings, models=models, responses=responses)
        )
        if args.publish_langsmith:
            publish_langsmith(cases, records, dataset_name=args.dataset_name)
        print(
            json.dumps(
                {
                    "mode": "provider" if args.provider else "offline",
                    "published": args.publish_langsmith,
                    "results": [
                        {key: record[key] for key in ("case_id", "model", "scores", "error_code")}
                        for record in records
                    ],
                },
                sort_keys=True,
            )
        )
        return 0 if all(all(record["scores"].values()) for record in records) else 1
    except Exception:
        # CLI never prints third-party errors which can embed prompts, outputs or API keys.
        print(
            "Evaluation failed: invalid input, configuration, or publication failure.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
