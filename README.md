# codex-llamacpp-proxy

Codex Desktop と llama.cpp `llama-server` の間に置く、Responses API 互換プロキシです。

## 背景

Windows 11 の Codex Desktop で `~/.codex/config.toml` の `profiles.local-llama` を使い、ローカルの llama.cpp に接続する検証を行いました。`profile = "local-llama"` や `codex app -c ...` によって Codex Desktop からローカルモデルへ接続できることは確認できました。

しかし、Codex Desktop から llama.cpp の OpenAI 互換エンドポイントへ直接接続すると、チャット開始時に次のエラーが発生しました。

```json
{"error":{"code":400,"message":"'type' of tool must be 'function'","type":"invalid_request_error"}}
```

原因は、Codex Desktop が `wire_api = "responses"` 前提で Responses API 形式の `tools` を送る一方で、llama.cpp 側の OpenAI 互換 API は Chat Completions 形式の `tools: [{"type":"function", ...}]` を期待するためです。

`web_search = "disabled"` や `image_generation = false`、各種 feature disable を試しても、Codex Desktop が送るリクエスト形式と llama.cpp が受け付ける形式の差は埋まりませんでした。また、`wire_api = "chat"` は Codex Desktop の config validation で弾かれました。

## 目的

`src/codex_llamacpp_proxy/proxy.py` はこの差を吸収するために作成した薄い変換プロキシです。

- Codex Desktop から `/v1/responses` として受ける
- llama.cpp の `/v1/chat/completions` に変換して転送する
- Responses API 形式の非 `function` tool を Chat Completions の `function` tool へ包み直す
- llama.cpp が返す `tool_calls` を Codex Desktop 側の Responses API `function_call` item へ戻す

これにより、少なくとも web search などの外部 tool を実際に使わない通常の問答は、Codex Desktop からローカル llama.cpp モデルに対して動作します。

## 使い方

llama.cpp `llama-server` が `http://127.0.0.1:8080/v1` で起動している前提で、プロキシを起動します。

```bash
uv run codex-llamacpp-proxy
```

別の llama.cpp URL を使う場合:

```bash
uv run codex-llamacpp-proxy --llama-base-url http://127.0.0.1:8080/v1
```

Codex の `~/.codex/config.toml` では、llama.cpp 直ではなくプロキシを `base_url` に指定します。

```toml
[model_providers.llamacpp-local]
name = "llama.cpp via local responses proxy"
base_url = "http://127.0.0.1:8090/v1"
wire_api = "responses"
requires_openai_auth = false
```

デバッグログを出したい場合:

```bash
PROXY_DEBUG=1 uv run codex-llamacpp-proxy
```

## 現状の制限

このプロキシは llama.cpp にリクエストを通すための実験的な変換層です。Codex Desktop 側の組み込み tool を完全に実行するものではありません。

特に `web_search` や `image_generation` は、llama.cpp が呼び出せる `function` tool の形には変換しますが、検索や画像生成そのものをプロキシ内で実行するわけではありません。通常の問答を通すことを第一目標にしています。

## 実装メモ

検証中に、単純な形式変換だけでは Codex Desktop 側のターンが正常終了しないケースが複数見つかりました。`src/codex_llamacpp_proxy/proxy.py` では次の互換処理も行っています。

### SSE の終了

一度は回答文字列が返っているのに Codex Desktop の `Working...` が継続する状態になりました。原因候補は、SSE 応答で `response.completed` と `[DONE]` を送っていても HTTP 接続が閉じられず、Codex Desktop 側がストリーム終了を確定できないことでした。

そのため、Responses API の SSE 応答では `Connection: close` を返し、`handler.close_connection = True` を設定しています。

### assistant prefill の回避

Qwen 系モデルでは、llama.cpp 側で次のエラーが出ることがありました。

```json
{"error":{"code":400,"message":"Assistant response prefill is incompatible with enable_thinking.","type":"invalid_request_error"}}
```

Codex Desktop の Responses リクエストには、前回の assistant 応答が履歴として含まれることがあります。これを Chat Completions の `messages` にそのまま変換すると、末尾の `assistant` メッセージが llama.cpp/Qwen の chat template で assistant prefill と解釈され、`enable_thinking` と衝突します。

そのため、Chat Completions へ転送する直前に、末尾の `assistant` メッセージを削る `strip_assistant_prefill()` を入れています。

### usage 形式の変換

Codex Desktop 側で次のような parse error が発生しました。

```text
stream disconnected before completion: failed to parse ResponseCompleted: missing field `input_tokens`
```

llama.cpp の Chat Completions usage は `prompt_tokens` / `completion_tokens` / `total_tokens` ですが、Responses API 側では `input_tokens` / `output_tokens` / `total_tokens` が期待されます。

そのため、`response.completed` に含める `usage` は `responses_usage_from_chat_usage()` で Responses API 形式へ変換しています。llama.cpp が usage を返さない場合も、Codex Desktop が parse できるよう 0 埋めの usage を返します。
