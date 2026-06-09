"""JSON-RPC 1.0 client for btxd (solo mining)."""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class RpcError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"RPC error {code}: {message}")
        self.code = code
        self.message = message


class BtxRpcClient:
    def __init__(
        self,
        url: str = "http://127.0.0.1:19334",
        rpc_user: str = "",
        rpc_password: str = "",
        cookie_file: str = "",
        timeout: float = 60.0,
    ):
        self.url = url.rstrip("/") + "/"
        self.timeout = timeout
        self._msg_id = 0
        self._auth_header = self._build_auth(rpc_user, rpc_password, cookie_file)

    @staticmethod
    def _build_auth(user: str, password: str, cookie_file: str) -> str:
        if user and password:
            token = f"{user}:{password}"
        elif cookie_file:
            path = Path(cookie_file).expanduser()
            token = path.read_text().strip()
        else:
            default_cookie = Path.home() / ".btx" / ".cookie"
            if default_cookie.is_file():
                token = default_cookie.read_text().strip()
            else:
                raise RuntimeError(
                    "no RPC credentials: set rpc_user/rpc_password or rpc_cookie_file"
                )
        return "Basic " + base64.b64encode(token.encode()).decode()

    def call(self, method: str, params: list | None = None, timeout: float | None = None) -> Any:
        self._msg_id += 1
        payload = json.dumps({
            "jsonrpc": "1.0",
            "id": self._msg_id,
            "method": method,
            "params": params or [],
        }).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": self._auth_header,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(f"RPC HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"RPC connection failed: {e}") from e

        if body.get("error"):
            err = body["error"]
            raise RpcError(err.get("code", -1), err.get("message", str(err)))
        return body.get("result")