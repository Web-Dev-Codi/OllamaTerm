from __future__ import annotations

import base64

from .base import Attachment, ParamsSchema, Tool, ToolContext, ToolResult

MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class WebFetchParams(ParamsSchema):
    url: str
    format: str = "markdown"  # "markdown" | "text" | "html"
    timeout: float | None = None  # seconds; max 120


class WebFetchTool(Tool):
    id = "webfetch"
    params_schema = WebFetchParams

    async def execute(self, params: WebFetchParams, ctx: ToolContext) -> ToolResult:
        url = params.url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            return ToolResult(
                title="webfetch", output="Invalid URL scheme.", metadata={"ok": False}
            )

        await ctx.ask(
            permission="webfetch",
            patterns=[url],
            always=["*"],
            metadata={"url": url},
        )

        timeout_sec = min(max(1.0, float(params.timeout or 30.0)), 120.0)
        try:
            import httpx  # noqa: WPS433
        except Exception as exc:  # pragma: no cover - optional dep
            return ToolResult(
                title="webfetch",
                output=f"Missing dependency: {exc}",
                metadata={"ok": False},
            )

        # Optional deps for HTML conversion — degrade gracefully when absent.
        try:
            from bs4 import BeautifulSoup  # type: ignore # noqa: WPS433

            _bs4_available = True
        except Exception:  # noqa: BLE001
            _bs4_available = False

        try:
            from markdownify import markdownify  # type: ignore # noqa: WPS433

            _markdownify_available = True
        except Exception:  # noqa: BLE001
            _markdownify_available = False

        fmt = (params.format or "markdown").strip().lower()
        if fmt not in {"markdown", "text", "html"}:
            fmt = "markdown"

        headers_map = {
            "markdown": "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1",
            "text": "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1",
            "html": "text/html;q=1.0, application/xhtml+xml;q=0.9, text/plain;q=0.8, */*;q=0.1",
        }
        headers = {
            "Accept": headers_map[fmt],
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        async with httpx.AsyncClient(
            timeout=timeout_sec, follow_redirects=True
        ) as client:
            response = await client.get(url, headers=headers)
            if (
                response.status_code == 403
                and response.headers.get("cf-mitigated") == "challenge"
            ):
                response = await client.get(
                    url, headers={**headers, "User-Agent": "ollamterm"}
                )

        size_header = response.headers.get("content-length")
        if size_header and int(size_header) > MAX_RESPONSE_BYTES:
            return ToolResult(
                title="webfetch", output="Response too large.", metadata={"ok": False}
            )

        data = await response.aread()
        if len(data) > MAX_RESPONSE_BYTES:
            return ToolResult(
                title="webfetch", output="Response too large.", metadata={"ok": False}
            )

        content_type = (
            response.headers.get("content-type", "").split(";")[0].strip().lower()
        )
        if (
            content_type.startswith("image/")
            and not content_type.endswith("svg+xml")
            and "vnd.fastbidsheet" not in content_type
        ):
            b64 = base64.b64encode(data).decode()
            att = Attachment(
                type="file", mime=content_type, url=f"data:{content_type};base64,{b64}"
            )
            return ToolResult(
                title="webfetch",
                output="Image fetched successfully.",
                metadata={"attachment": True},
                attachments=[att],
            )

        text = data.decode("utf-8", errors="replace")
        is_html = content_type in {"text/html", "application/xhtml+xml"}
        if fmt == "html" or not content_type:
            body = text
        elif fmt == "markdown" and is_html:
            if _markdownify_available:
                body = markdownify(
                    text,
                    heading_style="atx",
                    bullets="-",
                    strip=["script", "style", "meta", "link"],
                )
            else:
                body = text  # fall back to raw HTML
        elif fmt == "text" and is_html:
            if _bs4_available:
                body = BeautifulSoup(text, "html.parser").get_text(separator="\n")
            else:
                body = text  # fall back to raw HTML
        else:
            body = text

        return ToolResult(title="webfetch", output=body, metadata={})
