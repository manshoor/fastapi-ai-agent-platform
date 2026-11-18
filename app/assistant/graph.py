"""The agent loop as a LangGraph state graph.

    START -> model --tool call--> tools --> model ... --reply or stop--> END

The graph holds no data. Each turn passes its `TurnRun` in `config["configurable"]`;
nothing is checkpointed. Budgets are checked before every provider request:
at most `assistant_max_model_calls` model calls, and no call starts unless its full
timeout fits in the time left.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.ai.contracts import AssistantReply
from app.ai.provider import ToolCall
from app.assistant.config import AssistantSettings
from app.assistant.contracts import ActionResult, ToolName
from app.assistant.tools import TOOL_SCHEMAS, ToolContext, execute, output_text

BUDGET_CODE = "turn_budget"


@dataclass
class TurnRun:
    """Mutable state of one turn. Owned by the turn, never shared."""

    context: ToolContext
    model: object
    deadline: float
    items: list[dict]
    actions: list[ActionResult] = field(default_factory=list)
    reply: AssistantReply | None = None
    stop_code: str | None = None
    pending_call: ToolCall | None = None
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    response_model: str | None = None

    def remaining(self) -> float:
        return self.deadline - time.monotonic()


class _State(TypedDict):
    route: str


def _run(config: RunnableConfig) -> TurnRun:
    return config["configurable"]["turn"]


def build_graph(settings: AssistantSettings):
    provider_timeout = settings.ai_provider_timeout_seconds
    embedding_timeout = settings.ai_embedding_timeout_seconds

    async def model(state: _State, config: RunnableConfig) -> _State:
        run = _run(config)
        if run.model_calls >= settings.assistant_max_model_calls:
            run.stop_code = BUDGET_CODE
            return {"route": END}
        if run.remaining() < provider_timeout:
            run.stop_code = BUDGET_CODE
            return {"route": END}
        run.model_calls += 1
        step = await run.model.step(run.items, TOOL_SCHEMAS)
        run.response_model = step.model
        run.input_tokens += step.input_tokens or 0
        run.output_tokens += step.output_tokens or 0
        if step.reply is not None:
            run.reply = step.reply
            return {"route": END}
        call = step.call
        run.items.append(
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
        )
        run.pending_call = call
        return {"route": "tools"}

    async def tools(state: _State, config: RunnableConfig) -> _State:
        run = _run(config)
        call, run.pending_call = run.pending_call, None
        # No tool starts past the budget; a search must also fit its embedding call.
        needed = embedding_timeout if call.name == ToolName.SEARCH_DOCUMENTS else 0
        if run.remaining() <= needed:
            run.stop_code = BUDGET_CODE
            return {"route": END}
        outcome = await execute(run.context, call)
        run.actions.append(outcome.action)
        run.items.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output_text(outcome),
            }
        )
        return {"route": "model"}

    def route(state: _State) -> Literal["model", "tools", "__end__"]:
        return state["route"]

    graph = StateGraph(_State)
    graph.add_node("model", model)
    graph.add_node("tools", tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, ["tools", END])
    graph.add_conditional_edges("tools", route, ["model", END])
    return graph.compile()


def recursion_limit(settings: AssistantSettings) -> int:
    # Each model call may be followed by one tool node, plus the final step.
    return 2 * settings.assistant_max_model_calls + 2


def first_input(resident: dict, message: str) -> dict:
    """The resident's context and message as data, never as instructions."""
    return {"role": "user", "content": json.dumps({"resident": resident, "message": message})}
