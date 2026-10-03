# SPDX-License-Identifier: Apache-2.0
"""Wire protocol for disaggregated DFlash2 speculative decoding.

Format: every message is two ZMQ frames:
  frame 0 – msgpack-encoded header dict
  frame 1 – concatenated raw tensor bytes (may be empty)

Tensor bytes are always float16 (hidden states) or int32/int64 as noted.
"""

from __future__ import annotations

import struct
from typing import Any

import msgpack
import numpy as np
import torch

# ── message types ──────────────────────────────────────────────────────────────
MSG_PING    = 0  # health-check
MSG_PREFILL = 1  # new sequence, store context KV
MSG_DECODE  = 2  # batch decode step → draft tokens
MSG_FREE    = 3  # sequence finished, release KV blocks
MSG_ACK     = 4  # generic ok response
MSG_ERROR   = 5  # error response


# ── serialisation helpers ──────────────────────────────────────────────────────

def pack_tensor(t: torch.Tensor) -> bytes:
    """Return raw bytes for a contiguous CPU tensor."""
    return t.contiguous().numpy().tobytes()


def unpack_tensor(buf: bytes, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
    np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.int32:   np.int32,
        torch.int64:   np.int64,
    }[dtype]
    arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape)
    return torch.from_numpy(arr.copy())


# ── request builders ───────────────────────────────────────────────────────────

def build_ping(seq: int = 0) -> tuple[bytes, bytes]:
    return msgpack.packb({"t": MSG_PING, "seq": seq}), b""


def build_prefill(
    seq_id: str,
    hidden_states: torch.Tensor,   # [T, H] float16
    positions: torch.Tensor,        # [T] int64
    seq: int = 0,
) -> tuple[bytes, bytes]:
    T, H = hidden_states.shape
    header = {
        "t":      MSG_PREFILL,
        "seq_id": seq_id,
        "T":      T,
        "H":      H,
        "seq":    seq,
    }
    # payload: hidden_states bytes || positions bytes
    payload = pack_tensor(hidden_states.cpu().to(torch.float16)) + \
              pack_tensor(positions.cpu().to(torch.int64))
    return msgpack.packb(header), payload


def build_decode(
    seq_ids: list[str],
    hidden_states: torch.Tensor,   # [B, H] float16 — one new token per seq
    positions: torch.Tensor,        # [B] int64
    temperatures: torch.Tensor,     # [B] float32
    seeds: torch.Tensor,            # [B] int64
    bonus_token_ids: torch.Tensor,  # [B] int32 — actual token IDs for bonus (j=0)
    seq: int = 0,
) -> tuple[bytes, bytes]:
    B, H = hidden_states.shape
    header = {
        "t":       MSG_DECODE,
        "seq_ids": seq_ids,
        "B":       B,
        "H":       H,
        "seq":     seq,
    }
    payload = (
        pack_tensor(hidden_states.cpu().to(torch.float16)) +
        pack_tensor(positions.cpu().to(torch.int64)) +
        pack_tensor(temperatures.cpu().to(torch.float32)) +
        pack_tensor(seeds.cpu().to(torch.int64)) +
        pack_tensor(bonus_token_ids.cpu().to(torch.int32))
    )
    return msgpack.packb(header), payload


def build_free(seq_id: str, seq: int = 0) -> tuple[bytes, bytes]:
    return msgpack.packb({"t": MSG_FREE, "seq_id": seq_id, "seq": seq}), b""


# ── response builders ──────────────────────────────────────────────────────────

def build_ack(seq: int = 0) -> tuple[bytes, bytes]:
    return msgpack.packb({"t": MSG_ACK, "seq": seq}), b""


def build_draft_response(draft_tokens: torch.Tensor, seq: int = 0) -> tuple[bytes, bytes]:
    # draft_tokens: [B, K] int32
    B, K = draft_tokens.shape
    header = {"t": MSG_ACK, "B": B, "K": K, "seq": seq}
    return msgpack.packb(header), pack_tensor(draft_tokens.cpu().to(torch.int32))


def build_error(msg: str, seq: int = 0) -> tuple[bytes, bytes]:
    return msgpack.packb({"t": MSG_ERROR, "msg": msg, "seq": seq}), b""


# ── response parsers ───────────────────────────────────────────────────────────

def parse_header(frame: bytes) -> dict:
    return msgpack.unpackb(frame, raw=False)


def parse_draft_response(
    header: dict, payload: bytes
) -> torch.Tensor:
    B, K = header["B"], header["K"]
    return unpack_tensor(payload, torch.int32, (B, K))


def parse_prefill_payload(
    header: dict, payload: bytes
) -> tuple[torch.Tensor, torch.Tensor]:
    T, H = header["T"], header["H"]
    hs_bytes = T * H * 2          # float16
    pos_bytes = T * 8              # int64
    hidden_states = unpack_tensor(payload[:hs_bytes], torch.float16, (T, H))
    positions     = unpack_tensor(payload[hs_bytes:hs_bytes + pos_bytes],
                                  torch.int64, (T,))
    return hidden_states, positions


def parse_decode_payload(
    header: dict, payload: bytes
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H = header["B"], header["H"]
    hs_bytes    = B * H * 2   # float16
    pos_bytes   = B * 8       # int64
    temp_bytes  = B * 4       # float32
    seed_bytes  = B * 8       # int64
    bonus_bytes = B * 4       # int32
    o = 0
    hidden_states = unpack_tensor(payload[o:o+hs_bytes],   torch.float16, (B, H)); o += hs_bytes
    positions     = unpack_tensor(payload[o:o+pos_bytes],  torch.int64,   (B,));   o += pos_bytes
    temperatures  = unpack_tensor(payload[o:o+temp_bytes], torch.float32, (B,));   o += temp_bytes
    seeds         = unpack_tensor(payload[o:o+seed_bytes], torch.int64,   (B,));   o += seed_bytes
    bonus_ids     = unpack_tensor(payload[o:o+bonus_bytes],torch.int32,   (B,))
    return hidden_states, positions, temperatures, seeds, bonus_ids
