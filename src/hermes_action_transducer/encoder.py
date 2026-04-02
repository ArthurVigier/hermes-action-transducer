from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from hermes_action_transducer.models import HermesEncoder, HermesState, RobotObservation, RobotProfileSpec


class SimpleHermesEncoder(HermesEncoder):
    """
    Placeholder encoder.

    This keeps the architecture explicit now so we can later swap in:
    - hidden-state extraction from Hermes
    - pooled activations
    - a learned projection head
    """

    def encode(self, observation: RobotObservation, profile: RobotProfileSpec) -> HermesState:
        task = observation.task.strip()
        state = observation.state_text.strip() or "no state"
        summary = (
            f"profile={profile.name}; task={task}; state={state}; "
            f"tools={','.join(profile.runtime_tools[:4])}"
        )
        thought_vector = _string_to_vector(f"{task}|{state}|{profile.name}", size=8)
        intent_vector = _string_to_vector(task, size=8)
        return HermesState(
            summary_text=summary,
            thought_vector=thought_vector,
            intent_vector=intent_vector,
            metadata={
                "profile": profile.name,
                "task_length": len(task.split()),
                "has_image": observation.image_path is not None,
            },
        )


@dataclass
class HermesHFConfig:
    model_id: str = "NousResearch/Hermes-4.3-36B"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_length: int = 1024
    layer_index: int = -1
    pool_strategy: str = "mean"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None


class HermesHFEncoder(HermesEncoder):
    """
    Real Hermes encoder backed by Hugging Face Transformers.

    This extracts hidden states from a chosen layer and compresses them into
    compact thought/intent vectors for the downstream transducer.
    """

    def __init__(self, config: HermesHFConfig | None = None) -> None:
        self.config = config or HermesHFConfig()
        self._tokenizer = None
        self._model = None

    def encode(self, observation: RobotObservation, profile: RobotProfileSpec) -> HermesState:
        tokenizer, model = self._load_components()
        prompt = self._build_prompt(observation, profile)
        tokenized = self._tokenize(tokenizer, prompt)
        outputs = model(**tokenized, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        selected = hidden_states[self.config.layer_index]
        pooled = self._pool_hidden(selected, tokenized.get("attention_mask"))
        thought_vector = _compress_vector(pooled, size=8)
        intent_vector = _compress_vector(pooled[-64:] if len(pooled) >= 64 else pooled, size=8)
        return HermesState(
            summary_text=prompt[:400],
            thought_vector=thought_vector,
            intent_vector=intent_vector,
            metadata={
                "profile": profile.name,
                "model_id": self.config.model_id,
                "encoder_backend": "hermes_hf",
                "layer_index": self.config.layer_index,
                "pool_strategy": self.config.pool_strategy,
                "input_tokens": int(tokenized["input_ids"].shape[-1]),
            },
        )

    def _load_components(self):
        if self._tokenizer is None or self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Install HF encoder deps first: pip install -e '.[hermes]'"
                ) from exc

            dtype = self.config.torch_dtype
            if dtype == "auto":
                torch_dtype = "auto"
            else:
                torch_dtype = getattr(torch, dtype)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
            )
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": self.config.trust_remote_code,
                "device_map": self.config.device_map,
                "torch_dtype": torch_dtype,
            }
            if self.config.attn_implementation:
                model_kwargs["attn_implementation"] = self.config.attn_implementation
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs,
            )
            self._model.eval()
        return self._tokenizer, self._model

    def _build_prompt(self, observation: RobotObservation, profile: RobotProfileSpec) -> str:
        return (
            f"Robot profile: {profile.name}\n"
            f"Description: {profile.description}\n"
            f"Task: {observation.task}\n"
            f"State: {observation.state_text or '[none]'}\n"
            f"Proprio: {observation.proprio}\n"
            f"Runtime tools: {', '.join(profile.runtime_tools)}\n"
            "Return internal reasoning for the next best embodied action."
        )

    def _tokenize(self, tokenizer, prompt: str):
        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        if self._model is None:
            return tokenized
        device = getattr(self._model, "device", None)
        if device is None:
            return tokenized
        return {key: value.to(device) for key, value in tokenized.items()}

    def _pool_hidden(self, hidden_state, attention_mask):
        tensor = hidden_state[0]
        if self.config.pool_strategy == "last_token":
            last_index = _attention_length(attention_mask) - 1 if attention_mask is not None else -1
            return tensor[last_index].detach().float().cpu().tolist()

        if attention_mask is None:
            pooled = tensor.mean(dim=0)
            return pooled.detach().float().cpu().tolist()

        try:
            mask = attention_mask[0].unsqueeze(-1).to(tensor.dtype)
            pooled = (tensor * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)
            return pooled.detach().float().cpu().tolist()
        except Exception:
            pooled = tensor.mean(dim=0)
            return pooled.detach().float().cpu().tolist()


def _string_to_vector(raw: str, *, size: int) -> list[float]:
    values = [0.0] * size
    if not raw:
        return values
    for idx, ch in enumerate(raw.encode("utf-8")):
        values[idx % size] += ((ch % 31) / 30.0)
    return [round(value / max(1, len(raw)), 4) for value in values]


def _compress_vector(values: list[float], *, size: int) -> list[float]:
    if not values:
        return [0.0] * size
    out = [0.0] * size
    for idx, value in enumerate(values):
        out[idx % size] += float(value)
    scale = max(1, len(values) // size)
    return [round(value / scale, 4) for value in out]


def _attention_length(attention_mask) -> int:
    try:
        return int(attention_mask[0].sum().item())
    except Exception:
        try:
            values = attention_mask[0].tolist()
            return int(sum(values))
        except Exception:
            return 1
