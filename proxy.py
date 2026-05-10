#!/usr/bin/env python3
"""
Tiny Responses API -> llama.cpp Chat Completions proxy.

Run:
    uv run proxy.py

Then point Codex at:
    base_url = "http://127.0.0.1:8090/v1"
    wire_api = "responses"

Environment:
    PROXY_HOST              default: 127.0.0.1
    PROXY_PORT              default: 8090
    LLAMA_CPP_BASE_URL      default: http://127.0.0.1:8080/v1
    LLAMA_CPP_MODEL         optional fallback model name
    PROXY_DEBUG            set to 1 for request logging
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("PROXY_PORT", "8090"))
LLAMA_BASE_URL = os.environ.get("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
FALLBACK_MODEL = os.environ.get("LLAMA_CPP_MODEL")
DEBUG = os.environ.get("PROXY_DEBUG") == "1"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def debug(message: str) -> None:
    if DEBUG:
        log(message)


def now_unix() -> int:
    return int(time.time())


def response_id() -> str:
    return "resp_" + uuid.uuid4().hex


def output_id() -> str:
    return "msg_" + uuid.uuid4().hex


def call_id() -> str:
    return "call_" + uuid.uuid4().hex


def error_payload(message: str, status: int = 400, code: str = "proxy_error") -> bytes:
    return json.dumps(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "code": code,
            }
        },
        ensure_ascii=False,
    ).encode("utf-8")


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def send_error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    body = error_payload(message, status)
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def sse_frame(event: str, data: Any) -> bytes:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {text}\n\n".encode("utf-8")


def sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def text_from_content_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""

    part_type = part.get("type")
    if part_type in {"input_text", "output_text", "text"}:
        return str(part.get("text") or "")
    if part_type in {"input_image", "image_url"}:
        return "[image omitted by local proxy]"
    if part_type in {"file", "input_file"}:
        name = part.get("filename") or part.get("file_id") or "file"
        return f"[file omitted by local proxy: {name}]"
    if part_type in {"function_call_output", "tool_result"}:
        return str(part.get("output") or part.get("content") or "")
    return str(part.get("text") or part.get("content") or "")


def normalize_role(role: str | None) -> str:
    if role in {"system", "user", "assistant", "tool"}:
        return role
    if role == "developer":
        return "system"
    return "user"


def input_item_to_message(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        return {"role": "user", "content": item}
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    if item_type == "message" or "role" in item:
        role = normalize_role(item.get("role"))
        content = item.get("content", "")
        if isinstance(content, list):
            content = "\n".join(filter(None, (text_from_content_part(part) for part in content)))
        elif not isinstance(content, str):
            content = text_from_content_part(content)
        message: dict[str, Any] = {"role": role, "content": content}
        if role == "tool" and item.get("call_id"):
            message["tool_call_id"] = item["call_id"]
        return message

    if item_type in {"function_call_output", "tool_result"}:
        content = str(item.get("output") or item.get("content") or "")
        message = {"role": "tool", "content": content}
        if item.get("call_id"):
            message["tool_call_id"] = item["call_id"]
        return message

    if item_type in {"input_text", "text"}:
        return {"role": "user", "content": str(item.get("text") or "")}

    return {"role": "user", "content": text_from_content_part(item)}


def responses_input_to_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    input_value = payload.get("input", "")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        for item in input_value:
            message = input_item_to_message(item)
            if message is not None:
                messages.append(message)
    elif isinstance(input_value, dict):
        message = input_item_to_message(input_value)
        if message is not None:
            messages.append(message)

    if not messages:
        messages.append({"role": "user", "content": ""})
    messages = strip_assistant_prefill(messages)
    return messages


def strip_assistant_prefill(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Avoid llama.cpp/Qwen treating a trailing assistant message as prefill.

    Codex Responses requests may include prior assistant output items in the
    input list. Qwen chat templates with enable_thinking reject assistant
    prefill, so never forward a Chat Completions request ending in assistant.
    """
    if not messages:
        return messages

    stripped = list(messages)
    while len(stripped) > 1 and stripped[-1].get("role") == "assistant":
        removed = stripped.pop()
        debug(f"dropping trailing assistant prefill: {str(removed.get('content') or '')[:120]!r}")

    if stripped and stripped[-1].get("role") == "assistant":
        content = str(stripped[-1].get("content") or "")
        stripped[-1] = {
            "role": "user",
            "content": (
                "Previous assistant response, preserved as context rather than "
                f"assistant prefill:\n{content}"
            ),
        }

    return stripped


def convert_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") != "function":
        return wrap_responses_tool_as_function(tool)

    if isinstance(tool.get("function"), dict):
        function = dict(tool["function"])
    else:
        function = {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {},
        }

    if not function.get("name"):
        return None
    if not isinstance(function.get("parameters"), dict):
        function["parameters"] = {}
    return {"type": "function", "function": function}


def sanitize_function_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "tool")).strip("_")
    if not name:
        name = "tool"
    if len(name) > 64:
        name = name[:64].rstrip("_-") or "tool"
    return name


def default_parameters_for_responses_tool(tool_type: str) -> dict[str, Any]:
    if "web_search" in tool_type or tool_type in {"search", "browser_search"}:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": True,
        }
    if "image_generation" in tool_type or "image" in tool_type:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Image generation prompt.",
                }
            },
            "required": ["prompt"],
            "additionalProperties": True,
        }
    if "computer" in tool_type:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Computer action to perform.",
                }
            },
            "required": ["action"],
            "additionalProperties": True,
        }
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }


def wrap_responses_tool_as_function(tool: dict[str, Any]) -> dict[str, Any] | None:
    tool_type = str(tool.get("type") or "tool")
    name = sanitize_function_name(tool.get("name") or tool_type)
    parameters = tool.get("parameters") or tool.get("input_schema") or tool.get("schema")
    if not isinstance(parameters, dict):
        parameters = default_parameters_for_responses_tool(tool_type)

    description = tool.get("description")
    if not isinstance(description, str) or not description.strip():
        description = (
            f"Proxy wrapper for the Responses API tool '{tool_type}'. "
            "Call this function when that tool is needed."
        )

    debug(f"rewriting Responses tool {tool_type!r} as function {name!r}")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def convert_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted = [converted for tool in tools if (converted := convert_tool(tool)) is not None]
    debug(f"tools: received={len(tools)} forwarded={len(converted)}")
    return converted


def convert_tool_choice(choice: Any, tools: list[dict[str, Any]]) -> Any:
    if not tools:
        return None
    if choice in (None, "auto", "none", "required"):
        return choice
    if isinstance(choice, dict):
        if choice.get("type") == "function":
            if isinstance(choice.get("function"), dict):
                return choice
            if choice.get("name"):
                return {"type": "function", "function": {"name": choice["name"]}}
        if choice.get("type") in {"auto", "none", "required"}:
            return choice["type"]
    return "auto"


def responses_to_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model") or FALLBACK_MODEL
    if not model:
        raise ValueError("missing model; set model in Codex or LLAMA_CPP_MODEL")

    chat: dict[str, Any] = {
        "model": model,
        "messages": responses_input_to_messages(payload),
        "stream": bool(payload.get("stream")),
    }

    passthrough = [
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
    ]
    for key in passthrough:
        if key in payload and payload[key] is not None:
            chat[key] = payload[key]

    if "max_output_tokens" in payload and payload["max_output_tokens"] is not None:
        chat["max_tokens"] = payload["max_output_tokens"]

    tools = convert_tools(payload.get("tools"))
    if tools:
        chat["tools"] = tools
        tool_choice = convert_tool_choice(payload.get("tool_choice"), tools)
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice

    return chat


def chat_message_to_output_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (text_from_content_part(part) for part in content)))
    return "" if content is None else str(content)


def normalize_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False)


def chat_tool_calls_to_response_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return items

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = function.get("name") or tool_call.get("name")
        if not name:
            continue
        items.append(
            {
                "id": "fc_" + uuid.uuid4().hex,
                "type": "function_call",
                "call_id": str(tool_call.get("id") or call_id()),
                "name": str(name),
                "arguments": normalize_tool_arguments(function.get("arguments") or tool_call.get("arguments")),
            }
        )
    return items


def responses_usage_from_chat_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}

    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": int(input_details.get("cached_tokens") or usage.get("prompt_tokens_cached") or 0),
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        },
        "total_tokens": total_tokens,
    }


def responses_payload_from_chat(chat_payload: dict[str, Any], model: str, rid: str | None = None) -> dict[str, Any]:
    rid = rid or response_id()
    choice = (chat_payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = chat_message_to_output_text(message)
    output_items = chat_tool_calls_to_response_items(message)
    if text or not output_items:
        oid = output_id()
        output_items.insert(
            0,
            {
                "id": oid,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        )
    created = chat_payload.get("created") or now_unix()
    usage = responses_usage_from_chat_usage(chat_payload.get("usage"))

    return {
        "id": rid,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output_items,
        "parallel_tool_calls": True,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "usage": usage,
    }


def stream_response_object(handler: BaseHTTPRequestHandler, response: dict[str, Any]) -> None:
    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.send_header("cache-control", "no-cache")
    handler.send_header("connection", "close")
    handler.close_connection = True
    handler.end_headers()

    started = dict(response)
    started["status"] = "in_progress"
    started["output"] = []
    handler.wfile.write(sse_frame("response.created", {"type": "response.created", "response": started}))

    for index, item in enumerate(response.get("output") or []):
        added = dict(item)
        if added.get("type") == "message":
            added["status"] = "in_progress"
            added["content"] = []
        handler.wfile.write(
            sse_frame("response.output_item.added", {"type": "response.output_item.added", "output_index": index, "item": added})
        )

        if item.get("type") == "message":
            content = item.get("content") or []
            part = content[0] if content else {"type": "output_text", "text": "", "annotations": []}
            text = str(part.get("text") or "")
            handler.wfile.write(
                sse_frame(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": item.get("id"),
                        "output_index": index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
            )
            if text:
                handler.wfile.write(
                    sse_frame(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": item.get("id"),
                            "output_index": index,
                            "content_index": 0,
                            "delta": text,
                        },
                    )
                )
            handler.wfile.write(
                sse_frame(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": item.get("id"),
                        "output_index": index,
                        "content_index": 0,
                        "text": text,
                    },
                )
            )
            handler.wfile.write(
                sse_frame(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": item.get("id"),
                        "output_index": index,
                        "content_index": 0,
                        "part": part,
                    },
                )
            )

        handler.wfile.write(
            sse_frame("response.output_item.done", {"type": "response.output_item.done", "output_index": index, "item": item})
        )

    handler.wfile.write(sse_frame("response.completed", {"type": "response.completed", "response": response}))
    handler.wfile.write(sse_done())
    handler.wfile.flush()


def llama_request(path: str, payload: dict[str, Any], stream: bool) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        LLAMA_BASE_URL + path,
        data=body,
        headers={"content-type": "application/json", "accept": "text/event-stream" if stream else "application/json"},
        method="POST",
    )
    return urlopen(req, timeout=None)


def llama_get(path: str) -> tuple[int, bytes, str]:
    req = Request(LLAMA_BASE_URL + path, headers={"accept": "application/json"}, method="GET")
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.status, resp.read(), resp.headers.get("content-type", "application/json")
    except HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("content-type", "application/json")


def parse_sse_data(line: bytes) -> str | None:
    if not line.startswith(b"data:"):
        return None
    return line[5:].strip().decode("utf-8", errors="replace")


def stream_chat_as_responses(handler: BaseHTTPRequestHandler, upstream: Any, model: str) -> None:
    rid = response_id()
    oid = output_id()
    created = now_unix()
    full_text: list[str] = []

    handler.send_response(200)
    handler.send_header("content-type", "text/event-stream")
    handler.send_header("cache-control", "no-cache")
    handler.send_header("connection", "close")
    handler.close_connection = True
    handler.end_headers()

    def write(event: str, data: Any) -> None:
        handler.wfile.write(sse_frame(event, data))
        handler.wfile.flush()

    response_base = {
        "id": rid,
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "model": model,
        "output": [],
    }
    output_item = {
        "id": oid,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }

    write("response.created", {"type": "response.created", "response": response_base})
    write("response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": output_item})
    write(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": oid,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )

    for raw in upstream:
        data_text = parse_sse_data(raw)
        if data_text is None:
            continue
        if data_text == "[DONE]":
            break
        try:
            chunk = json.loads(data_text)
        except json.JSONDecodeError:
            debug(f"bad upstream SSE data: {data_text[:200]}")
            continue

        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        piece = delta.get("content") or ""
        if not piece:
            continue
        full_text.append(piece)
        write(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": oid,
                "output_index": 0,
                "content_index": 0,
                "delta": piece,
            },
        )

    text = "".join(full_text)
    write(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": oid,
            "output_index": 0,
            "content_index": 0,
            "text": text,
        },
    )
    write(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": oid,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        },
    )
    completed_item = {
        "id": oid,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    write("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": completed_item})
    completed_response = {
        **response_base,
        "status": "completed",
        "output": [completed_item],
        "usage": responses_usage_from_chat_usage(None),
    }
    write("response.completed", {"type": "response.completed", "response": completed_response})
    handler.wfile.write(sse_done())
    handler.wfile.flush()


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "CodexLlamaProxy/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        debug("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/health"}:
            send_json(self, 200, {"ok": True, "llama_base_url": LLAMA_BASE_URL})
            return
        if self.path == "/v1/models":
            status, body, content_type = llama_get("/models")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        send_error(self, 404, f"unknown endpoint: {self.path}")

    def do_POST(self) -> None:
        try:
            if self.path == "/v1/chat/completions":
                payload = read_json(self)
                stream = bool(payload.get("stream"))
                upstream = llama_request("/chat/completions", payload, stream)
                if stream:
                    self.send_response(upstream.status)
                    self.send_header("content-type", upstream.headers.get("content-type", "text/event-stream"))
                    self.end_headers()
                    for chunk in upstream:
                        self.wfile.write(chunk)
                    self.wfile.flush()
                else:
                    body = upstream.read()
                    self.send_response(upstream.status)
                    self.send_header("content-type", upstream.headers.get("content-type", "application/json"))
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return

            if self.path != "/v1/responses":
                send_error(self, 404, f"unknown endpoint: {self.path}")
                return

            payload = read_json(self)
            chat_payload = responses_to_chat_request(payload)
            debug(json.dumps({"chat_payload": chat_payload}, ensure_ascii=False)[:4000])

            client_wants_stream = bool(chat_payload.get("stream"))
            # Use a non-streaming upstream call for Responses requests so tool_calls
            # can be converted into complete Responses function_call items.
            chat_payload["stream"] = False
            upstream = llama_request("/chat/completions", chat_payload, False)
            chat_body = upstream.read()
            chat_json = json.loads(chat_body.decode("utf-8"))
            responses_json = responses_payload_from_chat(chat_json, chat_payload["model"])
            if client_wants_stream:
                stream_response_object(self, responses_json)
                return

            send_json(self, 200, responses_json)

        except HTTPError as exc:
            body = exc.read()
            self.send_response(exc.code)
            self.send_header("content-type", exc.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (URLError, ConnectionError) as exc:
            send_error(self, 502, f"failed to reach llama.cpp at {LLAMA_BASE_URL}: {exc}")
        except Exception as exc:
            if DEBUG:
                traceback.print_exc()
            send_error(self, 500, str(exc))


def main() -> int:
    global DEBUG, LLAMA_BASE_URL

    parser = argparse.ArgumentParser(description="Responses API -> llama.cpp Chat Completions proxy")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", default=PORT, type=int)
    parser.add_argument("--llama-base-url", default=LLAMA_BASE_URL)
    parser.add_argument("--debug", action="store_true", default=DEBUG)
    args = parser.parse_args()

    DEBUG = args.debug
    LLAMA_BASE_URL = args.llama_base_url.rstrip("/")

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    log(f"proxy listening on http://{args.host}:{args.port}/v1")
    log(f"forwarding to {LLAMA_BASE_URL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopping")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
