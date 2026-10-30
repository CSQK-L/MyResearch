---
sidebar_label: DeepSeek
---

# DeepSeek text chat (phase one)

Self-hosted Khoj supports DeepSeek as an independent chat provider using the
existing OpenAI SDK. It has separate credentials from OpenAI.

## Configure and start (WSL2 / Ubuntu)

From the repository root, activate your existing uv environment:

```bash
source .venv/bin/activate
export DEEPSEEK_API_KEY='your-deepseek-api-key'
# Optional: omit this variable to use the official endpoint.
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
```

Keep your existing PostgreSQL connection settings. For a fresh deployment, set
`KHOJ_ADMIN_EMAIL` and `KHOJ_ADMIN_PASSWORD` as usual, then start:

```bash
uv run --no-sync khoj --non-interactive
```

The normal startup migration/initialization flow registers `deepseek-flash` and
`deepseek-v4-pro` when it creates the initial chat configuration and a DeepSeek key
is configured. Without that key, no DeepSeek configuration is created. DeepSeek
creates no speech recognition or image generation models. Invalid DeepSeek
configuration stops initialization immediately and rolls back the initial chat
configuration writes rather than repeatedly creating other providers.

## Register models in an existing deployment

Back up your database using your usual deployment procedure. After installing
this version, review and apply its schema migration at your chosen maintenance
time, then run the additive registration command with your service's environment:

```bash
uv run --no-sync python src/khoj/manage.py migrate --plan
uv run --no-sync python src/khoj/manage.py migrate
uv run --no-sync python src/khoj/manage.py register_deepseek_models
```

The command reuses the `DeepSeek` AiModelApi record only when its key and base URL
match the environment. It is safe to repeat: it does not delete models, change
server defaults, or change user/Agent selections. Conflicting names, credentials,
provider types or model configurations produce an error and roll back the whole
registration. Resolve conflicts deliberately in `/server/admin` before retrying;
the command never rotates or overwrites credentials. API keys are not printed.
Restart Khoj with the startup command above after deploying the code.

`DEEPSEEK_BASE_URL` can point to a trusted OpenAI-compatible DeepSeek proxy.
Provider behavior uses the explicit `deepseek` type, rather than the model name
or endpoint hostname. The proxy receives your DeepSeek key.

## Select and verify

Open `http://localhost:42110/settings` and select `deepseek-flash` or
`deepseek-v4-pro` in the existing chat model selector. Models are registered with
`PriceTier.STANDARD`; existing subscription restrictions still apply. If needed,
configure your self-hosted subscription or deliberately select a server default
in the existing admin interface. Registration does not make that choice for you.

Start a new chat and send:

```text
/general 你好，请介绍一下自己
```

Check that text arrives incrementally and the full answer finishes normally.
The synchronous/internal request path also uses the existing OpenAI transport
and returns the normal `ResponseWithThought` object.

## Phase one scope

Requests use Chat Completions with `max_tokens` and explicitly disable thinking
with `extra_body={"thinking": {"type": "disabled"}}`. The `60000` token prompt
budget is a conservative Khoj configuration, **not** DeepSeek's official context
length. Unknown model names use Khoj's existing tokenizer fallback; no additional
tokenizer download is configured. Vision is disabled for both default models.

Only ordinary text chat and streaming are covered. JSON/schema output and tools
are not validated; Research multi-turn tools, thinking/reasoning persistence,
Responses API, images and other multimodal input are outside this phase. Using
`/general` avoids automatic tool selection. These limits describe this Khoj
integration, not the full capabilities of DeepSeek's models. Token costs are not
added to Khoj's cost table in this phase; its existing usage estimates may show
zero cost for these models.

Official references: [models and base URL](https://api-docs.deepseek.com/quick_start/pricing/),
[Chat Completions parameters](https://api-docs.deepseek.com/api/create-chat-completion/),
[thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/).
