import socket
import json
import time
from typing import Optional


class StratumClient:
    def __init__(self, host: str, port: int, payout_address: str, worker_name: str):
        self.host = host
        self.port = port
        self.payout_address = payout_address
        self.worker_name = worker_name
        self.sock: Optional[socket.socket] = None
        self._connect()

    def _connect(self):
        self.sock = socket.create_connection((self.host, self.port))
        self._send({"id": 1, "method": "mining.subscribe", "params": []})
        self._recv()
        self._send_authorize()

    def _send(self, msg: dict):
        line = json.dumps(msg) + "\n"
        self.sock.sendall(line.encode())

    def _recv(self) -> dict:
        buf = b""
        while b"\n" not in buf:
            buf += self.sock.recv(4096)
        return json.loads(buf.decode())

    def _send_authorize(self, address: str = None):
        addr = address or self.payout_address
        self._send({"id": 2, "method": "mining.authorize", "params": [addr, self.worker_name]})

    def send_authorize(self, address: str):
        self._send_authorize(address)

    def get_job(self) -> dict:
        while True:
            msg = self._recv()
            if msg.get("method") == "mining.notify":
                params = msg.get("params", {})
                return {
                    "prev_hash": params.get("prev_hash", "0" * 64),
                    "merkle_root": params.get("merkle_root", "0" * 64),
                    "time": int(params.get("time", 0)),
                    "bits": params.get("bits", "1d17c609"),
                    "seed_a": params.get("seed_a", "0" * 64),
                    "seed_b": params.get("seed_b", "0" * 64),
                    "block_height": int(params.get("block_height", 0)),
                }
            time.sleep(0.1)

    def submit_share(self, job: dict, result: dict):
        params = [
            self.payout_address,
            job["block_height"],
            result["nonce64"],
            job["prev_hash"],
            job["time"],
            job["bits"],
            job["merkle_root"],
        ]
        self._send({"id": 3, "method": "mining.submit", "params": params})