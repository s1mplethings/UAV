from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

LogFn = Callable[[str], None]

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_HTTP_USER_AGENT_ENV = "OPENAI_HTTP_USER_AGENT"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_HTTP_USER_AGENT = "OpenAI/Python 1.68.0"
RUNTIME_OPENAI_ENV_FILE = "openai.env"


@dataclass(frozen=True)
class OpenAIImageAnalysisResult:
    model: str
    image_path: str
    image_paths: tuple[str, ...]
    response_id: str | None
    text: str
    response_path: str
    api_base_url: str
    endpoint: str


def _runtime_env_candidates() -> tuple[Path, ...]:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        return (
            exe_dir / RUNTIME_OPENAI_ENV_FILE,
            exe_dir / "_internal" / "runtime_support" / RUNTIME_OPENAI_ENV_FILE,
        )
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root / ".runtime_secrets" / RUNTIME_OPENAI_ENV_FILE,
        repo_root / RUNTIME_OPENAI_ENV_FILE,
    )


def _parse_env_file(env_path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            payload[key] = value
    return payload


def load_openai_runtime_defaults() -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key in (OPENAI_API_KEY_ENV, OPENAI_BASE_URL_ENV, OPENAI_MODEL_ENV):
        env_value = os.environ.get(key, "").strip()
        if env_value:
            resolved[key] = env_value
    for env_path in _runtime_env_candidates():
        try:
            if env_path.exists():
                resolved.update(_parse_env_file(env_path))
                break
        except Exception:
            continue
    return resolved


def _image_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_output_text(response_json: dict[str, object]) -> str:
    top_level = response_json.get("output_text")
    if isinstance(top_level, str) and top_level.strip():
        return top_level.strip()

    chunks: list[str] = []
    for item in response_json.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n\n".join(chunks).strip()


def _extract_chat_text(response_json: dict[str, object]) -> str:
    choices = response_json.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n\n".join(chunks).strip()


def _normalize_base_url(base_url: str | None) -> str:
    resolved = (base_url or os.environ.get(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL)).strip()
    if not resolved:
        resolved = DEFAULT_OPENAI_BASE_URL
    return resolved.rstrip("/")


def _request_json(*, url: str, payload: dict[str, object], api_key: str) -> dict[str, object]:
    encoded_payload = json.dumps(payload).encode("utf-8")
    user_agent = os.environ.get(OPENAI_HTTP_USER_AGENT_ENV, DEFAULT_OPENAI_HTTP_USER_AGENT).strip()
    if not user_agent:
        user_agent = DEFAULT_OPENAI_HTTP_USER_AGENT
    request = urllib.request.Request(
        url,
        data=encoded_payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_http_error(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        detail_json = json.loads(body)
        return json.dumps(detail_json, ensure_ascii=False)
    except json.JSONDecodeError:
        return body or str(exc)


def _should_fallback_to_chat(exc: urllib.error.HTTPError, detail_message: str) -> bool:
    lowered = detail_message.lower()
    return exc.code in {400, 404, 405, 415, 422, 501} or any(
        pattern in lowered
        for pattern in (
            "responses",
            "not found",
            "unsupported",
            "not implemented",
            "unknown url",
            "no route",
            "chat/completions",
        )
    )


def _resolve_image_files(image_paths: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for image_path in image_paths:
        image_file = Path(image_path).expanduser().resolve()
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")
        resolved.append(image_file)
    if not resolved:
        raise ValueError("At least one image is required for OpenAI analysis.")
    return resolved


def analyze_images_with_openai(
    *,
    image_paths: Sequence[str],
    prompt: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4.1-mini",
    detail: str = "auto",
    response_path: str | None = None,
    log: LogFn = print,
) -> OpenAIImageAnalysisResult:
    runtime_defaults = load_openai_runtime_defaults()
    resolved_api_key = (api_key or runtime_defaults.get(OPENAI_API_KEY_ENV, "")).strip()
    if not resolved_api_key:
        raise ValueError(f"Missing OpenAI API key. Set {OPENAI_API_KEY_ENV} or provide it in the GUI.")
    if detail not in {"low", "high", "auto"}:
        raise ValueError("detail must be one of: low, high, auto")
    normalized_base_url = _normalize_base_url(base_url or runtime_defaults.get(OPENAI_BASE_URL_ENV))

    resolved_images = _resolve_image_files(image_paths)
    responses_content: list[dict[str, object]] = [{"type": "input_text", "text": prompt}]
    chat_content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for image_file in resolved_images:
        image_data_url = _image_data_url(image_file)
        responses_content.append({"type": "input_image", "image_url": image_data_url, "detail": detail})
        chat_content.append({"type": "image_url", "image_url": {"url": image_data_url, "detail": detail}})

    resolved_model = (model or runtime_defaults.get(OPENAI_MODEL_ENV) or "gpt-4.1-mini").strip()
    responses_payload = {
        "model": resolved_model,
        "input": [
            {
                "role": "user",
                "content": responses_content,
            }
        ],
    }
    chat_payload = {
        "model": resolved_model,
        "messages": [
            {
                "role": "user",
                "content": chat_content,
            }
        ],
    }

    endpoint_used = "responses"
    try:
        response_json = _request_json(
            url=f"{normalized_base_url}/responses",
            payload=responses_payload,
            api_key=resolved_api_key,
        )
    except urllib.error.HTTPError as exc:
        detail_message = _format_http_error(exc)
        if not _should_fallback_to_chat(exc, detail_message):
            raise RuntimeError(f"OpenAI API request failed ({exc.code}): {detail_message}") from exc
        log("[WARN] /responses unsupported on this base_url; falling back to /chat/completions")
        endpoint_used = "chat_completions"
        try:
            response_json = _request_json(
                url=f"{normalized_base_url}/chat/completions",
                payload=chat_payload,
                api_key=resolved_api_key,
            )
        except urllib.error.HTTPError as chat_exc:
            chat_detail = _format_http_error(chat_exc)
            raise RuntimeError(
                f"OpenAI-compatible API request failed on both /responses and /chat/completions: {chat_detail}"
            ) from chat_exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

    text = _extract_output_text(response_json) if endpoint_used == "responses" else _extract_chat_text(response_json)
    if not text:
        raise RuntimeError("OpenAI API returned no text content in the response.")

    output_path = (
        Path(response_path).expanduser().resolve()
        if response_path
        else resolved_images[0].with_name(resolved_images[0].stem + "_openai_response.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(response_json, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"[OK] OpenAI analysis response saved to: {output_path}")
    return OpenAIImageAnalysisResult(
        model=str(response_json.get("model", resolved_model)),
        image_path=str(resolved_images[0]),
        image_paths=tuple(str(path) for path in resolved_images),
        response_id=str(response_json.get("id")) if response_json.get("id") is not None else None,
        text=text,
        response_path=str(output_path),
        api_base_url=normalized_base_url,
        endpoint=endpoint_used,
    )


def analyze_image_with_openai(
    *,
    image_path: str,
    prompt: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "gpt-4.1-mini",
    detail: str = "auto",
    response_path: str | None = None,
    log: LogFn = print,
) -> OpenAIImageAnalysisResult:
    return analyze_images_with_openai(
        image_paths=[image_path],
        prompt=prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
        detail=detail,
        response_path=response_path,
        log=log,
    )


def analysis_result_to_dict(result: OpenAIImageAnalysisResult) -> dict[str, object]:
    return asdict(result)
