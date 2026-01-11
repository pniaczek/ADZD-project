#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# ============================
# CONFIG
# ============================
MCP_URL = "http://localhost:18888/mcp/"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"

MAX_CHARS_TO_LLM = 6000  # trochę więcej, ale nadal bezpiecznie na CPU
OLLAMA_NUM_PREDICT = 600
OLLAMA_TEMPERATURE = 0.2

RECENT_RUNS_PER_GROUP = 20  # pokaż do 20 runów danej nazwy


# ============================
# LLM
# ============================
async def ask_llm(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
        },
    }
    timeout = httpx.Timeout(600.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        return r.json().get("response", "")


# ============================
# MCP payload normalization
# ============================
def _extract_content(call_tool_result: Any) -> Any:
    return getattr(call_tool_result, "content", call_tool_result)


def _try_json_load(s: str) -> Optional[Any]:
    s2 = s.strip()
    if not s2:
        return None
    if s2.startswith("{") or s2.startswith("["):
        try:
            return json.loads(s2)
        except Exception:
            return None
    return None


def _to_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (dict, list, str, int, float, bool)):
        return obj

    # MCP TextContent-like
    if hasattr(obj, "text") and isinstance(getattr(obj, "text"), str):
        parsed = _try_json_load(obj.text)
        return parsed if parsed is not None else obj.text

    # pydantic
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    return str(obj)


def _parse_applications_payload(payload: Any) -> List[dict]:
    payload_plain = _to_plain(payload)

    if isinstance(payload_plain, list):
        apps: List[dict] = []
        for it in payload:
            it_plain = _to_plain(it)
            if isinstance(it_plain, dict):
                apps.append(it_plain)
            elif isinstance(it_plain, str):
                parsed = _try_json_load(it_plain)
                if isinstance(parsed, dict):
                    apps.append(parsed)
                elif isinstance(parsed, list):
                    apps.extend([x for x in parsed if isinstance(x, dict)])
            elif isinstance(it_plain, list):
                apps.extend([x for x in it_plain if isinstance(x, dict)])
        return apps

    if isinstance(payload_plain, dict):
        for key in ["applications", "apps", "items", "data", "result", "value"]:
            if key in payload_plain and isinstance(payload_plain[key], list):
                return [x for x in payload_plain[key] if isinstance(x, dict)]
        return [payload_plain]

    if isinstance(payload_plain, str):
        parsed = _try_json_load(payload_plain)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]

    raise RuntimeError(f"Could not parse applications from payload: {payload_plain}")


def _parse_single_object(payload: Any) -> Any:
    plain = _to_plain(payload)

    if isinstance(plain, dict):
        return plain

    if isinstance(plain, list):
        out: List[dict] = []
        for it in payload:
            it_plain = _to_plain(it)
            if isinstance(it_plain, dict):
                out.append(it_plain)
            elif isinstance(it_plain, str):
                parsed = _try_json_load(it_plain)
                if isinstance(parsed, dict):
                    out.append(parsed)
        if len(out) == 1:
            return out[0]
        if len(out) > 1:
            return {"items": out}
        return {"items": plain}

    if isinstance(plain, str):
        parsed = _try_json_load(plain)
        return parsed if parsed is not None else {"text": plain}

    return {"value": plain}


def _pick_app_id(app_obj: dict) -> Optional[str]:
    return (
        app_obj.get("id")
        or app_obj.get("appId")
        or app_obj.get("applicationId")
        or app_obj.get("app_id")
    )


def _pick_app_name(app_obj: dict) -> str:
    return str(app_obj.get("name") or app_obj.get("appName") or "UNKNOWN")


def _json_trunc(obj: Any, max_chars: int) -> str:
    s = json.dumps(obj, indent=2, default=str)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...<truncated>..."


# ============================
# UX helpers
# ============================
def _print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _ask_int(prompt: str, default: int, min_v: int, max_v: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        if min_v <= v <= max_v:
            return v
    except Exception:
        pass
    print("Niepoprawna wartość. Używam domyślnej.")
    return default


def _ask_yes_no(prompt: str, default_yes: bool = False) -> bool:
    default = "Y" if default_yes else "N"
    raw = input(f"{prompt} [y/N] (domyślnie {default}): ").strip().lower()
    if not raw:
        return default_yes
    return raw in ("y", "yes", "tak", "t")


def _prompt_multiline() -> str:
    print("\nWklej własny prompt (pusta linia kończy):\n")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


# ============================
# Prompts (5 + custom)
# ============================
PROMPTS: Dict[int, Dict[str, str]] = {
    1: {
        "title": "Szybki audyt",
        "text": (
            "You are a Spark performance analyst.\n"
            "Summarize what the application did, list any errors/warnings,\n"
            "and propose 3 concrete improvements.\n"
            "If the context lacks evidence for a claim, explicitly say 'not observed in context'."
        ),
    },
    2: {
        "title": "Bottlenecks i tuning",
        "text": (
            "You are a Spark performance analyst.\n"
            "Identify top bottlenecks (CPU/memory/shuffle/IO/skew) with evidence.\n"
            "Then propose concrete Spark config + code/data layout changes.\n"
            "If something is not present in context, say 'not observed in context'.\n"
            "Finish with a short experiment plan for the next run."
        ),
    },
    3: {
        "title": "Konfiguracja i środowisko (prod readiness)",
        "text": (
            "You are a Spark production readiness reviewer.\n"
            "Analyze environment/executor configuration and highlight risks.\n"
            "Propose a safe baseline configuration.\n"
            "Only use what's present in the context; otherwise say 'not observed in context'."
        ),
    },
    4: {
        "title": "SQL (slow SQL queries)",
        "text": (
            "You are a Spark SQL specialist.\n"
            "Analyze slow SQL queries / plans.\n"
            "Point out plan-level issues and suggest improvements.\n"
            "Only use evidence from context; otherwise say 'not observed in context'."
        ),
    },
    5: {
        "title": "Własny prompt",
        "text": "",
    },
}


# ============================
# Tool mapping: prompt -> tools (AUTO, no user selection)
# ============================
def tools_for_prompt(prompt_id: int) -> List[str]:
    # Zestawy są celowo proste, spójne i wystarczające.
    if prompt_id == 1:  # quick audit
        return [
            "get_application",
            "get_environment",
            "get_executor_summary",
            "list_slowest_stages",
            "get_job_bottlenecks",
        ]
    if prompt_id == 2:  # bottlenecks
        return [
            "get_application",
            "list_slowest_jobs",
            "list_slowest_stages",
            "get_job_bottlenecks",
            "get_resource_usage_timeline",
            "get_executor_summary",
            "get_environment",
        ]
    if prompt_id == 3:  # env
        return [
            "get_application",
            "get_environment",
            "list_executors",
            "get_executor_summary",
        ]
    if prompt_id == 4:  # SQL
        return [
            "get_application",
            "list_slowest_sql_queries",
            "compare_sql_execution_plans",
            "get_environment",
        ]
    # custom -> default to bottlenecks tools (najbardziej użyteczne)
    return tools_for_prompt(2)


# ============================
# Auto context collection
# ============================
async def _call_tool(session: ClientSession, tool_name: str, args: dict) -> Any:
    res = await session.call_tool(tool_name, args)
    payload = _extract_content(res)
    return _parse_single_object(payload)


async def build_context(session: ClientSession, app_id: str, tools: List[str], tools_available: List[str]) -> dict:
    ctx: Dict[str, Any] = {
        "app_id": app_id,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "tools_requested": tools,
    }

    for t in tools:
        if t not in tools_available:
            ctx[t] = {"error": "tool not available on server"}
            continue

        args = {"app_id": app_id}
        if t in ("list_slowest_jobs", "list_slowest_stages", "list_slowest_sql_queries"):
            args["limit"] = 10

        try:
            ctx[t] = await _call_tool(session, t, args)
        except Exception as e:
            ctx[t] = {"error": str(e), "args": args}

    return ctx


# ============================
# App grouping
# ============================
@dataclass
class AppEntry:
    app_id: str
    name: str
    raw: dict


def group_by_name(apps: List[dict]) -> Dict[str, List[AppEntry]]:
    grouped: Dict[str, List[AppEntry]] = {}
    for a in apps:
        app_id = _pick_app_id(a)
        if not app_id:
            continue
        name = _pick_app_name(a)
        grouped.setdefault(name, []).append(AppEntry(app_id=app_id, name=name, raw=a))
    return grouped


def select_app_name(grouped: Dict[str, List[AppEntry]]) -> str:
    names = list(grouped.keys())
    _print_header("1) Wybierz nazwę aplikacji (App Name)")
    for i, n in enumerate(names, start=1):
        print(f"{i}) {n}  (runs: {len(grouped[n])})")

    idx = _ask_int("Wybierz numer", default=1, min_v=1, max_v=len(names))
    return names[idx - 1]


def select_run(entries: List[AppEntry]) -> str:
    _print_header("2) Wybierz run tej aplikacji")
    show_n = min(len(entries), RECENT_RUNS_PER_GROUP)
    for i in range(show_n):
        print(f"{i+1}) {entries[i].app_id}")
    print("\n0) Najnowszy (domyślnie)")

    choice = _ask_int("Wybierz numer runu", default=0, min_v=0, max_v=show_n)
    if choice == 0:
        return entries[0].app_id
    return entries[choice - 1].app_id


def select_prompt() -> (int, str):
    _print_header("3) Wybierz rodzaj promptu")
    for i in range(1, 6):
        print(f"{i}) {PROMPTS[i]['title']}")

    pid = _ask_int("Wybierz numer", default=2, min_v=1, max_v=5)
    if pid == 5:
        custom = _prompt_multiline()
        if not custom:
            # fallback
            pid = 2
            return pid, PROMPTS[pid]["text"]
        return 5, custom
    return pid, PROMPTS[pid]["text"]


# ============================
# main
# ============================
async def main():
    async with streamablehttp_client(MCP_URL) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tools_available = [t.name for t in tools.tools]

            _print_header("MCP Spark Agent — tools dostępne w serwerze (informacyjnie)")
            for name in tools_available:
                print("-", name)

            # list applications
            apps_res = await session.call_tool("list_applications", {})
            apps_list = _parse_applications_payload(_extract_content(apps_res))
            if not apps_list:
                print("Brak aplikacji w history server.")
                return

            grouped = group_by_name(apps_list)
            if not grouped:
                print("Nie udało się zgrupować aplikacji (brak app_id?).")
                return

            app_name = select_app_name(grouped)
            app_id = select_run(grouped[app_name])

            prompt_id, prompt_text = select_prompt()
            selected_tools = tools_for_prompt(prompt_id)

            _print_header("Zbieranie kontekstu (automatycznie)")
            print(f"App name: {app_name}")
            print(f"Run id:   {app_id}")
            print(f"Prompt:   {PROMPTS[prompt_id]['title'] if prompt_id in PROMPTS else 'custom'}")
            print(f"Tools:    {selected_tools}")

            ctx = await build_context(session, app_id, selected_tools, tools_available)

            _print_header("Analiza LLM")
            ctx_json = _json_trunc(ctx, MAX_CHARS_TO_LLM)

            final_prompt = f"""
{prompt_text}

---
CONTEXT (truncated to {MAX_CHARS_TO_LLM} chars):
{ctx_json}

---
Return:
1) Summary
2) Evidence-based findings (bottlenecks / issues)
3) Concrete fixes
4) Experiment plan
"""

            answer = await ask_llm(final_prompt)

            print("\nLLM OUTPUT:\n")
            print(answer if answer else "(empty)")

            _print_header("Zapis (opcjonalnie)")
            if _ask_yes_no("Zapisać kontekst i odpowiedź do plików?", default_yes=False):
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                out_ctx = f"agent_context_{app_name}_{ts}.json".replace("/", "_")
                out_txt = f"agent_answer_{app_name}_{ts}.txt".replace("/", "_")
                with open(out_ctx, "w", encoding="utf-8") as f:
                    json.dump(ctx, f, indent=2, default=str)
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(answer)
                print(f"Saved: {out_ctx}")
                print(f"Saved: {out_txt}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
