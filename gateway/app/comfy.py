"""Async ComfyUI client: upload, submit, websocket progress, history, view.

Communicates with ComfyUI purely over HTTP/WS (no shared volume needed):
  - POST /upload/image      put the input texture into ComfyUI's input dir
  - POST /prompt            queue an API-format graph
  - WS   /ws?clientId=...   live execution + progress events
  - GET  /history/{id}      final outputs
  - GET  /view?...          download an output image
"""
import asyncio
import json
import uuid
import httpx
import websockets


class ComfyClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.ws_url = self.base.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self.client_id = uuid.uuid4().hex
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def aclose(self):
        await self._http.aclose()

    async def object_info(self) -> dict:
        r = await self._http.get(f"{self.base}/object_info")
        r.raise_for_status()
        return r.json()

    @staticmethod
    def combo_options(oi: dict, node: str, field: str) -> list:
        # ComfyUI uses two encodings: old `[[opt,...], {meta}]` (list at [0]) and
        # new `["COMBO", {"options":[opt,...]}]` (options dict at [1]).
        try:
            f = oi[node]["input"]["required"][field]
        except (KeyError, TypeError):
            return []
        spec = f[0]
        if isinstance(spec, list):
            return spec
        if len(f) > 1 and isinstance(f[1], dict) and isinstance(f[1].get("options"), list):
            return f[1]["options"]
        if isinstance(spec, dict) and isinstance(spec.get("options"), list):
            return spec["options"]
        return []

    async def upload_image(self, data: bytes, filename: str) -> str:
        files = {"image": (filename, data, "application/octet-stream")}
        r = await self._http.post(f"{self.base}/upload/image", files=files,
                                  data={"overwrite": "true", "type": "input"})
        r.raise_for_status()
        return r.json()["name"]

    async def submit(self, prompt: dict) -> str:
        r = await self._http.post(f"{self.base}/prompt",
                                  json={"prompt": prompt, "client_id": self.client_id})
        if r.status_code != 200:
            try:
                body = r.json()
            except Exception:
                body = {"error": r.text}
            raise PromptError(body)
        return r.json()["prompt_id"]

    async def history(self, prompt_id: str) -> dict:
        r = await self._http.get(f"{self.base}/history/{prompt_id}")
        r.raise_for_status()
        return r.json()

    async def in_queue(self, prompt_id: str) -> bool:
        """True if ComfyUI still has this prompt running or pending."""
        try:
            r = await self._http.get(f"{self.base}/queue")
            r.raise_for_status()
            q = r.json()
            for bucket in ("queue_running", "queue_pending"):
                for entry in q.get(bucket, []):
                    # entry = [number, prompt_id, prompt, extra_data, outputs_to_execute]
                    if len(entry) > 1 and entry[1] == prompt_id:
                        return True
        except Exception:
            return True  # on error, assume still queued (don't falsely fail a job)
        return False

    async def fetch_view(self, filename: str, subfolder: str, ftype: str) -> bytes:
        r = await self._http.get(f"{self.base}/view",
                                 params={"filename": filename, "subfolder": subfolder, "type": ftype})
        r.raise_for_status()
        return r.content

    async def listen(self, on_event):
        """Reconnecting websocket loop. Calls on_event(dict) for each text message."""
        url = f"{self.ws_url}?clientId={self.client_id}"
        while True:
            try:
                async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue  # binary previews — ignore
                        try:
                            await on_event(json.loads(raw))
                        except Exception:
                            pass
            except Exception:
                await asyncio.sleep(2)  # reconnect


class PromptError(Exception):
    def __init__(self, body):
        self.body = body
        super().__init__(str(body))
