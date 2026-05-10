"""Division coverage tests for codex_llamacpp_proxy.proxy.

Tests every branch/decision point in proxy.py. Network calls (urlopen) are
mocked via unittest.mock.patch so that no real HTTP connections are made.
"""

from __future__ import annotations

import json
import time
import threading
import http.server
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
from urllib.error import HTTPError, URLError

from codex_llamacpp_proxy.proxy import (
    # constants / module-level vars
    HOST,
    PORT,
    LLAMA_BASE_URL,
    DEBUG,
    # logging
    log,
    debug as debug_fn,
    # time / id helpers
    now_unix,
    response_id,
    output_id,
    call_id,
    # payload helpers
    error_payload,
    read_json,
    send_json,
    send_error,
    sse_frame,
    sse_done,
    # content extraction
    text_from_content_part,
    normalize_role,
    input_item_to_message,
    responses_input_to_messages,
    strip_assistant_prefill,
    # tool conversion
    convert_tool,
    sanitize_function_name,
    default_parameters_for_responses_tool,
    wrap_responses_tool_as_function,
    convert_tools,
    convert_tool_choice,
    # request / response conversion
    responses_to_chat_request,
    chat_message_to_output_text,
    normalize_tool_arguments,
    chat_tool_calls_to_response_items,
    responses_usage_from_chat_usage,
    responses_payload_from_chat,
    # streaming helpers
    stream_response_object,
    llama_request,
    llama_get,
    parse_sse_data,
    stream_chat_as_responses,
    # HTTP handler
    ProxyHandler,
    # CLI entry point
    main,
)


# ─── Fixtures / helpers ────────────────────────────────────────────────────────


def _make_handler(
    method: str = "GET",
    path: str = "/v1/health",
    body: bytes = b"",
    headers: dict | None = None,
) -> MagicMock:
    """Return a mock BaseHTTPRequestHandler that can be used with proxy functions."""
    handler = MagicMock(spec=BaseHTTPRequestHandler)
    handler.command = method
    handler.path = path

    merged_headers = {}
    if headers:
        merged_headers.update(headers)
    if body and "content-length" not in merged_headers:
        merged_headers["content-length"] = str(len(body))
    handler.headers = merged_headers

    def mock_send_response(status: int, message: str | None = None) -> None:
        handler.last_status = status

    handler.send_response = mock_send_response

    def mock_send_header(name: str, value: str) -> None:
        if not hasattr(handler, "headers_sent"):
            handler.headers_sent = {}
        handler.headers_sent[name] = value

    handler.send_header = mock_send_header
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    if body:
        handler.rfile = BytesIO(body)
    else:
        handler.rfile = BytesIO(b"")

    return handler


def _start_http_server(handler_cls, port=0):
    """Start a real HTTP server in a thread and return (server, local_url)."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, srv_port = server.server_address
    return server, f"http://{host}:{srv_port}"


# ─── Module-level constants ─────────────────────────────────────────────────────


class TestModuleConstants:
    def test_host_default(self):
        assert HOST == "127.0.0.1"

    def test_port_default(self):
        assert PORT == 8090

    def test_llama_base_url_default(self):
        assert LLAMA_BASE_URL == "http://127.0.0.1:8080/v1"

    def test_debug_default_false(self):
        assert DEBUG is False


# ─── Logging ─────────────────────────────────────────────────────────────────────


class TestLogDebug:
    @patch("codex_llamacpp_proxy.proxy.sys.stderr", new_callable=lambda: MagicMock())
    def test_log_writes_to_stderr(self, mock_stderr):
        log("test message")
        # print() adds a newline
        mock_stderr.write.assert_called()

    @patch("codex_llamacpp_proxy.proxy.DEBUG", True)
    @patch("codex_llamacpp_proxy.proxy.log")
    def test_debug_when_enabled(self, mock_log):
        debug_fn("debug msg")
        mock_log.assert_called_once_with("debug msg")

    @patch("codex_llamacpp_proxy.proxy.DEBUG", False)
    @patch("codex_llamacpp_proxy.proxy.log")
    def test_debug_when_disabled(self, mock_log):
        debug_fn("debug msg")
        mock_log.assert_not_called()


# ─── Time / ID helpers ───────────────────────────────────────────────────────────


class TestTimeIdHelpers:
    def test_now_unix_returns_int(self):
        before = int(time.time())
        val = now_unix()
        after = int(time.time())
        assert isinstance(val, int)
        assert before <= val <= after

    def test_response_id_format(self):
        rid = response_id()
        assert rid.startswith("resp_")
        assert len(rid) > 5

    def test_output_id_format(self):
        oid = output_id()
        assert oid.startswith("msg_")
        assert len(oid) > 4

    def test_call_id_format(self):
        cid = call_id()
        assert cid.startswith("call_")
        assert len(cid) > 6

    def test_ids_are_unique(self):
        ids = {response_id(), output_id(), call_id()}
        assert len(ids) == 3


# ─── error_payload ───────────────────────────────────────────────────────────────


class TestErrorPayload:
    def test_default_400(self):
        payload = error_payload("bad request")
        data = json.loads(payload.decode("utf-8"))
        assert data["error"]["message"] == "bad request"
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == "proxy_error"

    def test_4xx_returns_invalid_request_error(self):
        payload = error_payload("unauthorized", status=401, code="auth_error")
        data = json.loads(payload.decode("utf-8"))
        assert data["error"]["type"] == "invalid_request_error"

    def test_5xx_returns_server_error(self):
        payload = error_payload("internal error", status=500, code="server_err")
        data = json.loads(payload.decode("utf-8"))
        assert data["error"]["type"] == "server_error"

    def test_utf8_content(self):
        payload = error_payload("エラー発生")
        data = json.loads(payload.decode("utf-8"))
        assert data["error"]["message"] == "エラー発生"


# ─── read_json ───────────────────────────────────────────────────────────────────


class TestReadJson:
    def test_reads_valid_json(self):
        handler = _make_handler("POST", body=b'{"key": "value"}')
        assert read_json(handler) == {"key": "value"}

    def test_empty_body_returns_empty_dict(self):
        handler = _make_handler("POST", body=b"")
        assert read_json(handler) == {}

    def test_missing_content_length_returns_empty_dict(self):
        handler = _make_handler("POST", body=b"")
        handler.headers = {}
        assert read_json(handler) == {}

    def test_zero_content_length(self):
        handler = _make_handler("POST", body=b"", headers={"content-length": "0"})
        assert read_json(handler) == {}

    def test_json_with_utf8(self):
        handler = _make_handler(
            "POST", body='{"msg": "\u3053\u3093\u306b\u3061\u306f"}'.encode("utf-8")
        )
        assert read_json(handler) == {"msg": "\u3053\u3093\u306b\u3061\u306f"}


# ─── send_json / send_error ─────────────────────────────────────────────────────


class TestSendJson:
    def test_send_json_writes_body(self):
        handler = _make_handler()
        send_json(handler, 200, {"ok": True})
        handler.wfile.write.assert_called()
        call_args = handler.wfile.write.call_args
        written = call_args[0][0] if call_args[0] else b""
        assert b'"ok"' in written

    def test_send_json_sets_headers(self):
        handler = _make_handler()
        send_json(handler, 201, {"data": [1, 2]})
        assert handler.headers_sent.get("content-type") == "application/json"
        assert handler.headers_sent.get("content-length") is not None

    def test_send_json_status(self):
        handler = _make_handler()
        send_json(handler, 201, {"created": True})
        assert handler.last_status == 201


class TestSendError:
    def test_send_error_400(self):
        handler = _make_handler()
        send_error(handler, 400, "bad request")
        assert handler.last_status == 400
        written = b"".join(c[0][0] for c in handler.wfile.write.call_args_list)
        assert '"type": "invalid_request_error"' in written.decode("utf-8")

    def test_send_error_500(self):
        handler = _make_handler()
        send_error(handler, 500, "server error")
        assert handler.last_status == 500
        written = b"".join(c[0][0] for c in handler.wfile.write.call_args_list)
        assert '"type": "server_error"' in written.decode("utf-8")


# ─── SSE helpers ─────────────────────────────────────────────────────────────────


class TestSseHelpers:
    def test_sse_frame(self):
        frame = sse_frame("test_event", {"key": "val"})
        assert frame.startswith(b"event: test_event\ndata:")
        assert b'"key"' in frame
        assert frame.endswith(b"\n\n")

    def test_sse_frame_compact_json(self):
        frame = sse_frame("e", {"a": 1, "b": 2})
        # separators=(",", ":") → compact JSON
        assert b'"a":1' in frame

    def test_sse_done(self):
        assert sse_done() == b"data: [DONE]\n\n"


# ─── text_from_content_part ─────────────────────────────────────────────────────


class TestTextFromContentPart:
    def test_string_direct(self):
        assert text_from_content_part("hello") == "hello"

    def test_non_dict_non_str_returns_empty(self):
        assert text_from_content_part(123) == ""
        assert text_from_content_part(None) == ""
        assert text_from_content_part([1, 2]) == ""

    def test_input_text(self):
        assert text_from_content_part({"type": "input_text", "text": "hi"}) == "hi"

    def test_output_text(self):
        assert text_from_content_part({"type": "output_text", "text": "bye"}) == "bye"

    def test_text_type(self):
        assert text_from_content_part({"type": "text", "text": "t"}) == "t"

    def test_input_image(self):
        assert (
            text_from_content_part({"type": "input_image"})
            == "[image omitted by local proxy]"
        )

    def test_image_url(self):
        assert (
            text_from_content_part({"type": "image_url"})
            == "[image omitted by local proxy]"
        )

    def test_file_no_filename(self):
        result = text_from_content_part({"type": "file", "file_id": "f1"})
        assert result == "[file omitted by local proxy: f1]"

    def test_file_with_filename(self):
        result = text_from_content_part({"type": "input_file", "filename": "doc.pdf"})
        assert result == "[file omitted by local proxy: doc.pdf]"

    def test_function_call_output(self):
        assert (
            text_from_content_part({"type": "function_call_output", "output": "ok"})
            == "ok"
        )

    def test_tool_result(self):
        assert (
            text_from_content_part({"type": "tool_result", "content": "result"})
            == "result"
        )

    def test_empty_text(self):
        assert text_from_content_part({"type": "input_text", "text": ""}) == ""

    def test_missing_text_key(self):
        assert text_from_content_part({"type": "input_text"}) == ""

    def test_unknown_type_falls_back_to_text(self):
        assert (
            text_from_content_part({"type": "custom", "text": "fallback"}) == "fallback"
        )

    def test_unknown_type_falls_back_to_content(self):
        assert (
            text_from_content_part({"type": "custom", "content": "fallback2"})
            == "fallback2"
        )

    def test_unknown_type_empty(self):
        assert text_from_content_part({"type": "custom"}) == ""


# ─── normalize_role ─────────────────────────────────────────────────────────────


class TestNormalizeRole:
    def test_system(self):
        assert normalize_role("system") == "system"

    def test_user(self):
        assert normalize_role("user") == "user"

    def test_assistant(self):
        assert normalize_role("assistant") == "assistant"

    def test_tool(self):
        assert normalize_role("tool") == "tool"

    def test_developer_becomes_system(self):
        assert normalize_role("developer") == "system"

    def test_unknown_becomes_user(self):
        assert normalize_role("invalid") == "user"

    def test_none_becomes_user(self):
        assert normalize_role(None) == "user"

    def test_empty_string(self):
        assert normalize_role("") == "user"

    def test_whitespace(self):
        assert normalize_role("  ") == "user"


# ─── input_item_to_message ──────────────────────────────────────────────────────


class TestInputItemToMessage:
    def test_string_input(self):
        assert input_item_to_message("hello") == {"role": "user", "content": "hello"}

    def test_non_dict_non_str_returns_none(self):
        assert input_item_to_message(42) is None
        assert input_item_to_message(None) is None

    def test_message_type_with_string_content(self):
        result = input_item_to_message(
            {"type": "message", "role": "user", "content": "c"}
        )
        assert result == {"role": "user", "content": "c"}

    def test_message_type_with_list_content(self):
        item = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "a"},
                {"type": "input_text", "text": "b"},
            ],
        }
        assert input_item_to_message(item) == {"role": "user", "content": "a\nb"}

    def test_message_type_with_non_str_content(self):
        item = {"type": "message", "role": "user", "content": 123}
        assert input_item_to_message(item) == {"role": "user", "content": ""}

    def test_message_role_normalize(self):
        result = input_item_to_message(
            {"type": "message", "role": "developer", "content": "s"}
        )
        assert result["role"] == "system"

    def test_message_with_tool_role_and_call_id(self):
        result = input_item_to_message(
            {"type": "message", "role": "tool", "content": "result", "call_id": "c1"}
        )
        assert result["tool_call_id"] == "c1"

    def test_function_call_output_type(self):
        result = input_item_to_message(
            {"type": "function_call_output", "output": "out", "call_id": "x"}
        )
        assert result["role"] == "tool"
        assert result["content"] == "out"
        assert result["tool_call_id"] == "x"

    def test_tool_result_type(self):
        result = input_item_to_message({"type": "tool_result", "content": "c"})
        assert result["role"] == "tool"

    def test_input_text_type(self):
        result = input_item_to_message({"type": "input_text", "text": "t"})
        assert result == {"role": "user", "content": "t"}

    def test_text_type(self):
        result = input_item_to_message({"type": "text", "text": "t2"})
        assert result == {"role": "user", "content": "t2"}

    def test_fallback_to_text_from_content_part(self):
        result = input_item_to_message({"type": "input_image", "url": "http://x"})
        assert result == {"role": "user", "content": "[image omitted by local proxy]"}

    def test_message_without_type_field(self):
        result = input_item_to_message({"role": "user", "content": "direct"})
        assert result == {"role": "user", "content": "direct"}

    def test_message_with_none_content(self):
        result = input_item_to_message(
            {"type": "message", "role": "user", "content": None}
        )
        assert result["content"] == ""

    def test_message_with_list_content_empty(self):
        result = input_item_to_message(
            {"type": "message", "role": "user", "content": []}
        )
        assert result["content"] == ""


# ─── responses_input_to_messages ────────────────────────────────────────────────


class TestResponsesInputToMessages:
    def test_minimal_empty_payload(self):
        result = responses_input_to_messages({})
        assert result == [{"role": "user", "content": ""}]

    def test_with_string_input(self):
        result = responses_input_to_messages({"input": "hello"})
        assert {"role": "user", "content": "hello"} in result

    def test_with_list_input(self):
        # strip_assistant_prefill removes trailing assistant messages
        result = responses_input_to_messages(
            {
                "input": [
                    {"type": "message", "role": "user", "content": "a"},
                    {"type": "message", "role": "assistant", "content": "b"},
                ]
            }
        )
        # assistant at end is stripped
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "a"

    def test_with_dict_input(self):
        result = responses_input_to_messages(
            {"input": {"type": "input_text", "text": "single"}}
        )
        assert {"role": "user", "content": "single"} in result

    def test_instructions_become_system(self):
        result = responses_input_to_messages(
            {"instructions": "be helpful", "input": "hi"}
        )
        assert result[0] == {"role": "system", "content": "be helpful"}

    def test_empty_instructions_not_added(self):
        result = responses_input_to_messages({"instructions": "", "input": "hi"})
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 0

    def test_whitespace_only_instructions_not_added(self):
        result = responses_input_to_messages({"instructions": "   ", "input": "hi"})
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 0

    def test_none_input(self):
        result = responses_input_to_messages({"input": None, "instructions": "sys"})
        # None input doesn't match str/list/dict, so no user message added
        # Only system message from instructions remains
        assert {"role": "system", "content": "sys"} in result

    def test_input_is_zero(self):
        result = responses_input_to_messages({"input": 0})
        assert {"role": "user", "content": ""} in result

    def test_input_is_empty_list(self):
        result = responses_input_to_messages({"input": []})
        assert {"role": "user", "content": ""} in result

    def test_input_is_empty_string(self):
        result = responses_input_to_messages({"input": ""})
        assert {"role": "user", "content": ""} in result


# ─── strip_assistant_prefill ────────────────────────────────────────────────────


class TestStripAssistantPrefill:
    def test_empty_list(self):
        assert strip_assistant_prefill([]) == []

    def test_single_assistant_converted_to_user(self):
        msgs = [{"role": "assistant", "content": "alone"}]
        result = strip_assistant_prefill(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Previous assistant response" in result[0]["content"]

    def test_multiple_trailing_assistants_stripped(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]
        result = strip_assistant_prefill(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_trailing_assistant_converted_to_user(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "last response"},
        ]
        result = strip_assistant_prefill(msgs)
        # while len > 1 and last is assistant → pops assistant
        # remaining: [{"role": "user", "content": "hi"}]
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hi"

    def test_no_trailing_assistant_unchanged(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "back"},
        ]
        result = strip_assistant_prefill(msgs)
        assert len(result) == 3

    def test_only_assistant_messages_keeps_last_as_user(self):
        msgs = [
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]
        result = strip_assistant_prefill(msgs)
        # All trailing assistants are stripped, but the last one is converted to user
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_assistant_with_empty_content(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ]
        result = strip_assistant_prefill(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_assistant_in_middle_preserved(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "mid"},
            {"role": "user", "content": "back"},
        ]
        result = strip_assistant_prefill(msgs)
        assert len(result) == 3
        assert result[1]["role"] == "assistant"


# ─── convert_tool ────────────────────────────────────────────────────────────────


class TestConvertTool:
    def test_non_dict_returns_none(self):
        assert convert_tool("string") is None
        assert convert_tool(42) is None

    def test_function_type_with_function_dict(self):
        tool = {"type": "function", "function": {"name": "calc", "parameters": {}}}
        result = convert_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "calc"

    def test_function_type_missing_function_key(self):
        tool = {"type": "function", "name": "n", "description": "d", "parameters": {}}
        result = convert_tool(tool)
        assert result["function"]["name"] == "n"

    def test_non_function_type_wrapped(self):
        tool = {"type": "web_search", "name": "search"}
        result = convert_tool(tool)
        assert result["type"] == "function"

    def test_function_without_name_returns_none(self):
        tool = {"type": "function", "function": {}}
        assert convert_tool(tool) is None

    def test_function_with_none_parameters(self):
        tool = {"type": "function", "function": {"name": "n", "parameters": None}}
        result = convert_tool(tool)
        assert result["function"]["parameters"] == {}

    def test_function_with_non_dict_parameters(self):
        tool = {"type": "function", "function": {"name": "n", "parameters": "invalid"}}
        result = convert_tool(tool)
        assert result["function"]["parameters"] == {}


# ─── sanitize_function_name ─────────────────────────────────────────────────────


class TestSanitizeFunctionName:
    def test_alphanumeric(self):
        assert sanitize_function_name("my_tool") == "my_tool"

    def test_spaces_become_underscores(self):
        assert sanitize_function_name("my tool") == "my_tool"

    def test_special_chars(self):
        assert sanitize_function_name("tool@#$name") == "tool_name"

    def test_empty_input(self):
        assert sanitize_function_name("") == "tool"

    def test_none_input(self):
        assert sanitize_function_name(None) == "tool"

    def test_truncate_long_name(self):
        long_name = "a" * 100
        result = sanitize_function_name(long_name)
        assert len(result) <= 64

    def test_truncate_removes_trailing_underscores(self):
        result = sanitize_function_name("a" * 70)
        assert len(result) == 64
        assert not result.endswith("_")

    def test_only_special_chars(self):
        assert sanitize_function_name("@#$%") == "tool"


# ─── default_parameters_for_responses_tool ──────────────────────────────────────


class TestDefaultParametersForResponsesTool:
    def test_web_search(self):
        params = default_parameters_for_responses_tool("web_search")
        assert "query" in params["properties"]
        assert "query" in params["required"]

    def test_search_type(self):
        params = default_parameters_for_responses_tool("search")
        assert "query" in params["properties"]

    def test_browser_search(self):
        params = default_parameters_for_responses_tool("browser_search")
        assert "query" in params["properties"]

    def test_image_generation(self):
        params = default_parameters_for_responses_tool("image_generation")
        assert "prompt" in params["properties"]

    def test_image_type(self):
        params = default_parameters_for_responses_tool("draw_image")
        assert "prompt" in params["properties"]

    def test_computer(self):
        params = default_parameters_for_responses_tool("computer_control")
        assert "action" in params["properties"]

    def test_generic_fallback(self):
        params = default_parameters_for_responses_tool("custom_tool")
        assert params["properties"] == {}


# ─── wrap_responses_tool_as_function ────────────────────────────────────────────


class TestWrapResponsesToolAsFunction:
    def test_with_name(self):
        tool = {"type": "web_search", "name": "search"}
        result = wrap_responses_tool_as_function(tool)
        assert result["function"]["name"] == "search"

    def test_without_name_uses_type(self):
        tool = {"type": "web_search"}
        result = wrap_responses_tool_as_function(tool)
        assert result["function"]["name"] == "web_search"

    def test_with_parameters(self):
        tool = {"type": "t", "parameters": {"type": "object", "properties": {}}}
        result = wrap_responses_tool_as_function(tool)
        assert result["function"]["parameters"]["type"] == "object"

    def test_with_input_schema(self):
        tool = {"type": "t", "input_schema": {"type": "object"}}
        result = wrap_responses_tool_as_function(tool)
        assert result["function"]["parameters"]["type"] == "object"

    def test_with_schema(self):
        tool = {"type": "t", "schema": {"type": "object"}}
        result = wrap_responses_tool_as_function(tool)
        assert result["function"]["parameters"]["type"] == "object"

    def test_missing_parameters_uses_default(self):
        tool = {"type": "web_search"}
        result = wrap_responses_tool_as_function(tool)
        assert "query" in result["function"]["parameters"]["properties"]

    def test_with_description(self):
        tool = {"type": "t", "description": "A search tool"}
        result = wrap_responses_tool_as_function(tool)
        assert result["function"]["description"] == "A search tool"

    def test_empty_description_gets_default(self):
        tool = {"type": "t", "description": "   "}
        result = wrap_responses_tool_as_function(tool)
        assert "Proxy wrapper" in result["function"]["description"]

    def test_no_description_gets_default(self):
        tool = {"type": "t"}
        result = wrap_responses_tool_as_function(tool)
        assert "Proxy wrapper" in result["function"]["description"]

    def test_empty_type(self):
        tool = {"type": ""}
        result = wrap_responses_tool_as_function(tool)
        assert result["function"]["name"] == "tool"


# ─── convert_tools ──────────────────────────────────────────────────────────────


class TestConvertTools:
    def test_non_list_returns_empty(self):
        assert convert_tools("string") == []
        assert convert_tools(None) == []
        assert convert_tools({}) == []

    def test_empty_list(self):
        assert convert_tools([]) == []

    def test_filters_none_results(self):
        tools = [
            {"type": "function", "function": {"name": "ok"}},
            {"type": "function", "function": {}},  # no name → None
        ]
        result = convert_tools(tools)
        assert len(result) == 1

    def test_all_valid(self):
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        result = convert_tools(tools)
        assert len(result) == 2


# ─── convert_tool_choice ────────────────────────────────────────────────────────


class TestConvertToolChoice:
    def test_no_tools_returns_none(self):
        assert convert_tool_choice("auto", []) is None

    def test_none_choice(self):
        assert convert_tool_choice(None, ["t"]) is None

    def test_auto(self):
        assert convert_tool_choice("auto", ["t"]) == "auto"

    def test_none(self):
        assert convert_tool_choice("none", ["t"]) == "none"

    def test_required(self):
        assert convert_tool_choice("required", ["t"]) == "required"

    def test_function_dict_with_function_key(self):
        choice = {"type": "function", "function": {"name": "my_func"}}
        assert convert_tool_choice(choice, ["t"]) == choice

    def test_function_dict_with_name_only(self):
        choice = {"type": "function", "name": "my_func"}
        result = convert_tool_choice(choice, ["t"])
        assert result == {"type": "function", "function": {"name": "my_func"}}

    def test_function_dict_no_name_returns_auto(self):
        choice = {"type": "function"}
        result = convert_tool_choice(choice, ["t"])
        assert result == "auto"

    def test_type_auto_none_required_as_string(self):
        for t in ("auto", "none", "required"):
            choice = {"type": t}
            assert convert_tool_choice(choice, ["t"]) == t

    def test_unknown_type_returns_auto(self):
        assert convert_tool_choice({"type": "unknown"}, ["t"]) == "auto"

    def test_string_choice_auto(self):
        assert convert_tool_choice("auto", ["t"]) == "auto"

    def test_string_choice_none(self):
        assert convert_tool_choice("none", ["t"]) == "none"

    def test_string_choice_required(self):
        assert convert_tool_choice("required", ["t"]) == "required"


# ─── responses_to_chat_request ─────────────────────────────────────────────────


class TestResponsesToChatRequest:
    def test_missing_model_raises(self):
        with patch("codex_llamacpp_proxy.proxy.FALLBACK_MODEL", None):
            with pytest.raises(ValueError, match="missing model"):
                responses_to_chat_request({})

    def test_with_fallback_model(self):
        with patch("codex_llamacpp_proxy.proxy.FALLBACK_MODEL", "llama-3"):
            result = responses_to_chat_request({"input": "hi"})
            assert result["model"] == "llama-3"

    def test_stream_true(self):
        result = responses_to_chat_request(
            {"model": "m", "input": "hi", "stream": True}
        )
        assert result["stream"] is True

    def test_stream_false(self):
        result = responses_to_chat_request(
            {"model": "m", "input": "hi", "stream": False}
        )
        assert result["stream"] is False

    def test_stream_missing(self):
        result = responses_to_chat_request({"model": "m", "input": "hi"})
        assert result["stream"] is False

    def test_passthrough_params(self):
        result = responses_to_chat_request(
            {
                "model": "m",
                "input": "hi",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                "seed": 42,
                "stop": ["\n"],
            }
        )
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["max_tokens"] == 100
        assert result["presence_penalty"] == 0.1
        assert result["frequency_penalty"] == 0.2
        assert result["seed"] == 42
        assert result["stop"] == ["\n"]

    def test_max_output_tokens_maps_to_max_tokens(self):
        result = responses_to_chat_request(
            {"model": "m", "input": "hi", "max_output_tokens": 256}
        )
        assert result["max_tokens"] == 256

    def test_max_output_tokens_none_not_mapped(self):
        result = responses_to_chat_request(
            {"model": "m", "input": "hi", "max_output_tokens": None}
        )
        assert "max_tokens" not in result or result.get("max_tokens") is None

    def test_with_tools(self):
        result = responses_to_chat_request(
            {
                "model": "m",
                "input": "hi",
                "tools": [{"type": "function", "function": {"name": "calc"}}],
            }
        )
        assert len(result["tools"]) == 1

    def test_with_tool_choice(self):
        result = responses_to_chat_request(
            {
                "model": "m",
                "input": "hi",
                "tools": [{"type": "function", "function": {"name": "calc"}}],
                "tool_choice": "auto",
            }
        )
        assert result["tool_choice"] == "auto"

    def test_model_from_payload_takes_precedence(self):
        with patch("codex_llamacpp_proxy.proxy.FALLBACK_MODEL", "fallback"):
            result = responses_to_chat_request({"model": "gpt-4", "input": "hi"})
            assert result["model"] == "gpt-4"


# ─── chat_message_to_output_text ────────────────────────────────────────────────


class TestChatMessageToOutputText:
    def test_string_content(self):
        assert chat_message_to_output_text({"content": "hello"}) == "hello"

    def test_list_content(self):
        msg = {
            "content": [
                {"type": "output_text", "text": "a"},
                {"type": "input_text", "text": "b"},
            ]
        }
        assert chat_message_to_output_text(msg) == "a\nb"

    def test_none_content(self):
        assert chat_message_to_output_text({"content": None}) == ""

    def test_missing_content(self):
        assert chat_message_to_output_text({}) == ""

    def test_non_string_non_list_non_none(self):
        assert chat_message_to_output_text({"content": 42}) == "42"


# ─── normalize_tool_arguments ──────────────────────────────────────────────────


class TestNormalizeToolArguments:
    def test_already_string(self):
        assert normalize_tool_arguments('{"key": "val"}') == '{"key": "val"}'

    def test_none_becomes_empty_object(self):
        assert normalize_tool_arguments(None) == "{}"

    def test_dict_to_json(self):
        assert normalize_tool_arguments({"key": "val"}) == '{"key": "val"}'

    def test_list_to_json(self):
        result = normalize_tool_arguments([1, 2])
        assert result == "[1, 2]"


# ─── chat_tool_calls_to_response_items ─────────────────────────────────────────


class TestChatToolCallsToResponseItems:
    def test_no_tool_calls(self):
        assert chat_tool_calls_to_response_items({"content": "hi"}) == []

    def test_non_list_tool_calls(self):
        assert chat_tool_calls_to_response_items({"tool_calls": "string"}) == []

    def test_empty_tool_calls(self):
        assert chat_tool_calls_to_response_items({"tool_calls": []}) == []

    def test_single_tool_call(self):
        msg = {
            "tool_calls": [
                {
                    "id": "tc1",
                    "function": {"name": "calc", "arguments": '{"x": 1}'},
                }
            ]
        }
        items = chat_tool_calls_to_response_items(msg)
        assert len(items) == 1
        assert items[0]["type"] == "function_call"
        assert items[0]["name"] == "calc"
        assert items[0]["arguments"] == '{"x": 1}'
        assert items[0]["call_id"] == "tc1"

    def test_tool_call_missing_function_uses_name(self):
        msg = {"tool_calls": [{"name": "n"}]}
        items = chat_tool_calls_to_response_items(msg)
        assert len(items) == 1
        assert items[0]["name"] == "n"

    def test_tool_call_with_string_arguments(self):
        msg = {"tool_calls": [{"function": {"name": "n", "arguments": "{}"}}]}
        items = chat_tool_calls_to_response_items(msg)
        assert items[0]["arguments"] == "{}"

    def test_non_dict_tool_call_skipped(self):
        msg = {"tool_calls": ["invalid"]}
        assert chat_tool_calls_to_response_items(msg) == []

    def test_tool_call_without_name_skipped(self):
        msg = {"tool_calls": [{"function": {}}]}
        assert chat_tool_calls_to_response_items(msg) == []

    def test_arguments_fallback_from_tool_call(self):
        msg = {"tool_calls": [{"name": "n", "arguments": '{"a":1}'}]}
        items = chat_tool_calls_to_response_items(msg)
        assert items[0]["arguments"] == '{"a":1}'

    def test_call_id_fallback(self):
        msg = {"tool_calls": [{"function": {"name": "n"}}]}
        items = chat_tool_calls_to_response_items(msg)
        assert items[0]["call_id"].startswith("call_")


# ─── responses_usage_from_chat_usage ───────────────────────────────────────────


class TestResponsesUsageFromChatUsage:
    def test_none_usage(self):
        result = responses_usage_from_chat_usage(None)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_empty_dict(self):
        result = responses_usage_from_chat_usage({})
        assert result["input_tokens"] == 0

    def test_with_prompt_tokens(self):
        result = responses_usage_from_chat_usage(
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20
        assert result["total_tokens"] == 30

    def test_with_input_output_tokens(self):
        result = responses_usage_from_chat_usage(
            {"input_tokens": 5, "output_tokens": 15, "total_tokens": 20}
        )
        assert result["input_tokens"] == 5
        assert result["output_tokens"] == 15

    def test_total_tokens_fallback(self):
        result = responses_usage_from_chat_usage(
            {"prompt_tokens": 10, "completion_tokens": 20}
        )
        assert result["total_tokens"] == 30

    def test_cached_tokens(self):
        result = responses_usage_from_chat_usage(
            {
                "prompt_tokens_cached": 5,
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }
        )
        assert result["input_tokens_details"]["cached_tokens"] == 5

    def test_reasoning_tokens(self):
        result = responses_usage_from_chat_usage(
            {
                "output_tokens_details": {"reasoning_tokens": 3},
                "prompt_tokens": 10,
                "completion_tokens": 20,
            }
        )
        assert result["output_tokens_details"]["reasoning_tokens"] == 3

    def test_input_tokens_details_as_dict(self):
        result = responses_usage_from_chat_usage({"prompt_tokens": 1})
        assert isinstance(result["input_tokens_details"], dict)
        assert "cached_tokens" in result["input_tokens_details"]

    def test_output_tokens_details_as_dict(self):
        result = responses_usage_from_chat_usage({"completion_tokens": 1})
        assert isinstance(result["output_tokens_details"], dict)
        assert "reasoning_tokens" in result["output_tokens_details"]


# ─── responses_payload_from_chat ───────────────────────────────────────────────


class TestResponsesPayloadFromChat:
    def test_basic_conversion(self):
        chat = {
            "choices": [{"message": {"content": "hello", "role": "assistant"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        result = responses_payload_from_chat(chat, "test-model")
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["model"] == "test-model"
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["role"] == "assistant"

    def test_with_tool_calls(self):
        chat = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "function": {"name": "calc", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = responses_payload_from_chat(chat, "m")
        func_items = [i for i in result["output"] if i["type"] == "function_call"]
        assert len(func_items) == 1
        # No text message should be inserted when there are tool calls
        text_items = [i for i in result["output"] if i["type"] == "message"]
        assert len(text_items) == 0

    def test_text_and_tool_calls(self):
        chat = {
            "choices": [
                {
                    "message": {
                        "content": "here is the answer",
                        "tool_calls": [{"function": {"name": "n"}}],
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = responses_payload_from_chat(chat, "m")
        text_items = [i for i in result["output"] if i["type"] == "message"]
        assert len(text_items) == 1

    def test_no_choices(self):
        chat = {
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        result = responses_payload_from_chat(chat, "m")
        assert len(result["output"]) == 1

    def test_custom_rid(self):
        chat = {"choices": [{"message": {"content": ""}}]}
        result = responses_payload_from_chat(chat, "m", rid="resp_custom")
        assert result["id"] == "resp_custom"

    def test_error_and_incomplete_null(self):
        chat = {"choices": [{"message": {"content": ""}}]}
        result = responses_payload_from_chat(chat, "m")
        assert result["error"] is None
        assert result["incomplete_details"] is None

    def test_parallel_tool_calls_true(self):
        chat = {"choices": [{"message": {"content": ""}}]}
        result = responses_payload_from_chat(chat, "m")
        assert result["parallel_tool_calls"] is True

    def test_default_model_field(self):
        chat = {"choices": [{"message": {"content": ""}}]}
        result = responses_payload_from_chat(chat, "m")
        assert result["model"] == "m"

    def test_empty_choices_list(self):
        chat = {
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        result = responses_payload_from_chat(chat, "m")
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"

    def test_choice_without_message(self):
        chat = {
            "choices": [{}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        result = responses_payload_from_chat(chat, "m")
        assert len(result["output"]) == 1

    def test_message_with_list_content(self):
        chat = {
            "choices": [
                {"message": {"content": [{"type": "output_text", "text": "multi"}]}}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        result = responses_payload_from_chat(chat, "m")
        text_item = result["output"][0]
        assert text_item["content"][0]["text"] == "multi"


# ─── parse_sse_data ────────────────────────────────────────────────────────────


class TestParseSseData:
    def test_data_line(self):
        assert parse_sse_data(b"data: hello") == "hello"

    def test_data_line_with_spaces(self):
        assert parse_sse_data(b"data:   hello  ") == "hello"

    def test_non_data_line(self):
        assert parse_sse_data(b"event: test") is None

    def test_empty_line(self):
        assert parse_sse_data(b"") is None

    def test_utf8_in_data(self):
        # Use actual UTF-8 bytes for "こんにちは"
        utf8_bytes = "こんにちは".encode("utf-8")
        result = parse_sse_data(b"data: " + utf8_bytes)
        assert result == "こんにちは"


# ─── stream_response_object ────────────────────────────────────────────────────


class TestStreamResponseObject:
    def test_writes_sse_frames(self):
        handler = _make_handler()
        response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1234,
            "status": "completed",
            "model": "m",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "hi", "annotations": []}
                    ],
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        stream_response_object(handler, response)

        calls = handler.wfile.write.call_args_list
        # Should write at least: response.created, output_item.added, content_part.added,
        # output_text.delta, output_text.done, content_part.done, output_item.done,
        # response.completed, [DONE]
        assert len(calls) >= 9

    def test_empty_output(self):
        handler = _make_handler()
        response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1234,
            "status": "completed",
            "model": "m",
            "output": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        stream_response_object(handler, response)
        calls = handler.wfile.write.call_args_list
        # Should still write response.created and response.completed
        assert len(calls) >= 2

    def test_no_output_field(self):
        handler = _make_handler()
        response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1234,
            "status": "completed",
            "model": "m",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        stream_response_object(handler, response)
        calls = handler.wfile.write.call_args_list
        assert len(calls) >= 2

    def test_message_with_empty_content(self):
        handler = _make_handler()
        response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1234,
            "status": "completed",
            "model": "m",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [],
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        stream_response_object(handler, response)
        calls = handler.wfile.write.call_args_list
        assert len(calls) >= 7

    def test_non_message_output_item(self):
        handler = _make_handler()
        response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1234,
            "status": "completed",
            "model": "m",
            "output": [{"id": "fc_1", "type": "function_call", "name": "test"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        stream_response_object(handler, response)
        calls = handler.wfile.write.call_args_list
        # Should NOT write content_part or output_text frames for non-message items
        all_written = b"".join(c[0][0] for c in calls)
        assert b"output_text" not in all_written

    def test_two_message_items(self):
        handler = _make_handler()
        response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1234,
            "status": "completed",
            "model": "m",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "first", "annotations": []}
                    ],
                },
                {
                    "id": "msg_2",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "second", "annotations": []}
                    ],
                },
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        stream_response_object(handler, response)

        calls = handler.wfile.write.call_args_list
        all_written = b"".join(c[0][0] for c in calls)
        # Both items should appear with different output_index values (compact JSON)
        assert b'"output_index":0' in all_written
        assert b'"output_index":1' in all_written


# ─── llama_request / llama_get ─────────────────────────────────────────────────


class TestLlamaRequest:
    @patch("codex_llamacpp_proxy.proxy.urlopen")
    def test_llama_request_posts_json(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_urlopen.return_value = mock_resp
        llama_request("/chat/completions", {"model": "m", "messages": []}, False)
        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        # Request.headers is a plain dict with capitalized keys
        assert (
            req.headers.get("Content-type") == "application/json"
            or req.headers.get("content-type") == "application/json"
        )

    @patch("codex_llamacpp_proxy.proxy.urlopen")
    def test_llama_request_stream_accept(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        llama_request("/chat/completions", {}, True)
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert (
            req.headers.get("Accept") == "text/event-stream"
            or req.headers.get("accept") == "text/event-stream"
        )

    @patch("codex_llamacpp_proxy.proxy.urlopen")
    def test_llama_request_non_stream_accept(self, mock_urlopen):
        mock_urlopen.return_value = MagicMock()
        llama_request("/chat/completions", {}, False)
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert (
            req.headers.get("Accept") == "application/json"
            or req.headers.get("accept") == "application/json"
        )


class TestLlamaGet:
    @patch("codex_llamacpp_proxy.proxy.urlopen")
    def test_llama_get_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"models": []}'
        mock_resp.headers.get.return_value = "application/json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        status, body, ct = llama_get("/models")
        assert status == 200
        assert body == b'{"models": []}'
        assert ct == "application/json"

    @patch("codex_llamacpp_proxy.proxy.urlopen")
    def test_llama_get_http_error(self, mock_urlopen):
        # Create a real HTTPError-like exception with required attributes
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"error"
        mock_exc = HTTPError(
            "http://localhost/models",
            500,
            "server error",
            {"content-type": "application/json"},
            mock_resp,
        )
        mock_urlopen.side_effect = mock_exc

        status, body, ct = llama_get("/models")
        assert status == 500
        assert body == b"error"
        assert ct == "application/json"


# ─── ProxyHandler — GET ────────────────────────────────────────────────────────


class TestProxyHandlerGet:
    @patch("codex_llamacpp_proxy.proxy.llama_get")
    def test_health(self, mock_llama_get):
        mock_llama_get.return_value = (200, b'{"ok": true}', "application/json")
        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read())
                assert data["ok"] is True
        finally:
            server.shutdown()

    @patch("codex_llamacpp_proxy.proxy.llama_get")
    def test_v1_health(self, mock_llama_get):
        mock_llama_get.return_value = (200, b'{"ok": true}', "application/json")
        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(f"{url}/v1/health")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
        finally:
            server.shutdown()

    @patch("codex_llamacpp_proxy.proxy.llama_get")
    def test_v1_models_proxies(self, mock_llama_get):
        mock_llama_get.return_value = (
            200,
            b'{"data": [{"id": "m"}]}',
            "application/json",
        )
        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(f"{url}/v1/models")
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read())
                assert "data" in data
        finally:
            server.shutdown()

    @patch("codex_llamacpp_proxy.proxy.llama_get")
    def test_unknown_endpoint_returns_404(self, mock_llama_get):
        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(f"{url}/unknown")
            with pytest.raises(Exception):
                urllib.request.urlopen(req)
        finally:
            server.shutdown()


# ─── ProxyHandler — POST /v1/chat/completions (passthrough) ────────────────────


class TestProxyHandlerChatCompletionsPassthrough:
    @patch("codex_llamacpp_proxy.proxy.llama_request")
    def test_non_stream_passthrough(self, mock_llama_request):
        mock_upstream = MagicMock()
        mock_upstream.status = 200
        mock_upstream.read.return_value = (
            b'{"choices": [{"message": {"content": "hi"}}]}'
        )
        mock_upstream.headers.get.return_value = "application/json"
        mock_llama_request.return_value = mock_upstream

        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=b'{"model": "m", "messages": []}',
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
        finally:
            server.shutdown()

    @patch("codex_llamacpp_proxy.proxy.llama_request")
    def test_stream_passthrough(self, mock_llama_request):
        mock_upstream = MagicMock()
        mock_upstream.status = 200
        mock_upstream.headers.get.return_value = "text/event-stream"
        mock_upstream.__iter__ = MagicMock(return_value=iter([b"data: {}"]))
        mock_llama_request.return_value = mock_upstream

        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{url}/v1/chat/completions",
                data=b'{"model": "m", "messages": [], "stream": true}',
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
        finally:
            server.shutdown()


# ─── ProxyHandler — POST /v1/responses ─────────────────────────────────────────


class TestProxyHandlerResponsesEndpoint:
    @patch("codex_llamacpp_proxy.proxy.llama_request")
    def test_basic_conversion_and_response(self, mock_llama_request):
        mock_upstream = MagicMock()
        mock_upstream.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "hello from llama"}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                },
            }
        ).encode("utf-8")
        mock_llama_request.return_value = mock_upstream

        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{url}/v1/responses",
                data=b'{"model": "m", "input": "hello"}',
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                data = json.loads(resp.read())
                assert data["object"] == "response"
                assert data["status"] == "completed"
                assert len(data["output"]) >= 1
        finally:
            server.shutdown()

    @patch("codex_llamacpp_proxy.proxy.llama_request")
    def test_streaming_responses(self, mock_llama_request):
        mock_upstream = MagicMock()
        mock_upstream.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "streamed"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 5,
                    "total_tokens": 6,
                },
            }
        ).encode("utf-8")
        mock_llama_request.return_value = mock_upstream

        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{url}/v1/responses",
                data=b'{"model": "m", "input": "hi", "stream": true}',
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.status == 200
                content_type = resp.headers.get("content-type")
                assert "event-stream" in content_type
        finally:
            server.shutdown()

    @patch("codex_llamacpp_proxy.proxy.llama_request")
    def test_unknown_post_endpoint(self, mock_llama_request):
        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{url}/v1/unknown",
                data=b"{}",
                headers={"content-type": "application/json"},
            )
            with pytest.raises(Exception):
                urllib.request.urlopen(req)
        finally:
            server.shutdown()

    @patch("codex_llamacpp_proxy.proxy.llama_request")
    def test_llama_http_error_propagated(self, mock_llama_request):
        mock_exc = HTTPError(
            "http://localhost/v1/chat/completions",
            500,
            "server error",
            {"content-type": "application/json"},
            MagicMock(),
        )
        mock_exc.read = MagicMock(return_value=b'{"error": "bad"}')
        mock_llama_request.side_effect = mock_exc

        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{url}/v1/responses",
                data=b'{"model": "m", "input": "hi"}',
                headers={"content-type": "application/json"},
            )
            with pytest.raises(Exception):
                urllib.request.urlopen(req)
        finally:
            server.shutdown()

    @patch("codex_llamacpp_proxy.proxy.llama_request")
    def test_llama_url_error_returns_502(self, mock_llama_request):
        mock_llama_request.side_effect = URLError("connection refused")

        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{url}/v1/responses",
                data=b'{"model": "m", "input": "hi"}',
                headers={"content-type": "application/json"},
            )
            try:
                urllib.request.urlopen(req)
            except urllib.error.HTTPError as e:
                assert e.code == 502
        finally:
            server.shutdown()

    @patch("codex_llamacpp_proxy.proxy.llama_request")
    def test_general_exception_returns_500(self, mock_llama_request):
        mock_llama_request.side_effect = ValueError("bad payload")

        server, url = _start_http_server(
            lambda *a, **kw: type("H", (ProxyHandler,), {})(*a, **kw)
        )
        try:
            import urllib.request

            req = urllib.request.Request(
                f"{url}/v1/responses",
                data=b'{"model": "m", "input": "hi"}',
                headers={"content-type": "application/json"},
            )
            try:
                urllib.request.urlopen(req)
            except urllib.error.HTTPError as e:
                assert e.code == 500
        finally:
            server.shutdown()


# ─── stream_chat_as_responses ──────────────────────────────────────────────────


class TestStreamChatAsResponses:
    def test_text_streaming(self):
        handler = _make_handler()
        upstream_data = [
            b'data: {"choices":[{"delta":{"content":"H"}}]}',
            b'data: {"choices":[{"delta":{"content":"i"}}]}',
            b"data: [DONE]",
        ]
        upstream = iter(upstream_data)
        stream_chat_as_responses(handler, upstream, "test-model")

        calls = handler.wfile.write.call_args_list
        all_written = b"".join(c[0][0] for c in calls)
        assert b"response.created" in all_written
        assert b"response.output_text.delta" in all_written
        assert b'"delta":"H"' in all_written
        assert b'"delta":"i"' in all_written
        assert b"response.completed" in all_written

    def test_empty_response(self):
        handler = _make_handler()
        upstream_data = [b'data: {"choices":[{"delta":{}}]}', b"data: [DONE]"]
        upstream = iter(upstream_data)
        stream_chat_as_responses(handler, upstream, "m")

        calls = handler.wfile.write.call_args_list
        all_written = b"".join(c[0][0] for c in calls)
        assert b"response.created" in all_written
        assert b"response.completed" in all_written

    def test_bad_sse_data_logged(self):
        handler = _make_handler()
        upstream_data = [b"data: not json at all", b"data: [DONE]"]
        upstream = iter(upstream_data)
        stream_chat_as_responses(handler, upstream, "m")

        calls = handler.wfile.write.call_args_list
        assert len(calls) >= 2  # created + completed at minimum

    def test_no_done_sentinel(self):
        handler = _make_handler()
        upstream_data = [b'data: {"choices":[{"delta":{"content":"x"}}]}']
        upstream = iter(upstream_data)
        stream_chat_as_responses(handler, upstream, "m")

        calls = handler.wfile.write.call_args_list
        all_written = b"".join(c[0][0] for c in calls)
        assert b"response.completed" in all_written


# ─── main (CLI) ────────────────────────────────────────────────────────────────


class TestMain:
    @patch("codex_llamacpp_proxy.proxy.ThreadingHTTPServer")
    @patch("codex_llamacpp_proxy.proxy.log")
    def test_main_starts_server(self, mock_log, mock_server_cls):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        with patch(
            "sys.argv",
            [
                "proxy",
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
                "--llama-base-url",
                "http://x:8080/v1",
                "--debug",
            ],
        ):
            result = main()

        assert result == 0
        mock_server_cls.assert_called_once()
        mock_server.serve_forever.assert_called_once()

    @patch("codex_llamacpp_proxy.proxy.ThreadingHTTPServer")
    def test_main_keyboard_interrupt(self, mock_server_cls):
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt()
        mock_server_cls.return_value = mock_server

        with patch("sys.argv", ["proxy"]):
            result = main()

        assert result == 0

    @patch("codex_llamacpp_proxy.proxy.ThreadingHTTPServer")
    def test_main_default_args(self, mock_server_cls):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        with patch("sys.argv", ["proxy"]):
            main()

        args = mock_server_cls.call_args[0][0]
        assert args[0] == "127.0.0.1"
        assert args[1] == 8090

    @patch("codex_llamacpp_proxy.proxy.ThreadingHTTPServer")
    def test_main_trims_trailing_slash(self, mock_server_cls):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        with patch("sys.argv", ["proxy", "--llama-base-url", "http://x:8080/v1/"]):
            main()

        # LLAMA_BASE_URL should be trimmed
        from codex_llamacpp_proxy.proxy import LLAMA_BASE_URL as current_url

        assert not current_url.endswith("/")


# ─── Edge case: error_payload ensure_ascii ─────────────────────────────────────


class TestErrorPayloadUtf8:
    def test_ensure_ascii_false(self):
        payload = error_payload("\u3053\u3093\u306b\u3061\u306f")
        # ensure_ascii=False means the raw UTF-8 bytes should be present
        assert b"\xe3\x81" in payload or "\\u3053" not in payload.decode("utf-8")
