"""Brain-Score ArtificialSubject for LLaDA-8B-Base."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly, NeuroidAssembly
from brainscore_language import ArtificialSubject
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
LAYER = 24


class LLaDA8BSubject(ArtificialSubject):
    """Clean LLaDA neural readout with passage-local context accumulation.

    The neural operating regime is fixed: an uncorrupted model input, layer 24,
    and the mean hidden state over tokens belonging to the current text part.
    Each ``digest_text`` call starts fresh; an ordered list accumulates context
    only within that call. Reading-time behavior uses prefix-only one-mask
    surprisal in bits with independent tokenizer boundaries.
    """

    def __init__(self, *, model_path: str | Path | None = None) -> None:
        resolved_path = model_path or os.environ.get("LLADA_MODEL_PATH") or MODEL_ID
        load_kwargs = {"trust_remote_code": True}
        if isinstance(resolved_path, Path) or Path(str(resolved_path)).is_dir():
            load_kwargs["local_files_only"] = True
        else:
            load_kwargs["revision"] = MODEL_REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(str(resolved_path), **load_kwargs)
        model_kwargs = {
            **load_kwargs,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        # LLaDA-8B fits on one 24GB GPU.  Respect the executor's visible-device
        # allocation instead of allowing Accelerate to spread a benchmark job
        # across unrelated GPUs on a shared host.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(str(resolved_path), **model_kwargs)
        self.model.to(device)
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        self._recording: tuple[str, str] | None = None
        self._behavioral_task: str | None = None
        self._cache: dict[tuple[str, tuple[int, int]], np.ndarray] = {}

    def identifier(self) -> str:
        return "llada-8b-base"

    def start_behavioral_task(self, task: ArtificialSubject.Task) -> None:
        if task != ArtificialSubject.Task.reading_times:
            raise NotImplementedError(f"unsupported behavioral task: {task}")
        self._behavioral_task = task

    def _part_surprisal(self, parts: list[str]) -> tuple[float, str]:
        normalized = [part.strip() for part in parts]
        if not normalized:
            raise ValueError("behavioral text parts must not be empty")
        context = re.sub(r"\s+([.,!?;:])", r"\1",
                         " ".join(part for part in normalized if part))
        if not normalized[-1]:
            return 0.0, context
        current = re.sub(r"\s+([.,!?;:])", r"\1", normalized[-1])
        span = (len(context) - len(current), len(context))
        # Tokenizing the completed context can change the prefix IDs at BPE
        # boundaries.  Freeze the available prefix before scoring the target.
        prefix_text = context[:span[0]].rstrip()
        continuation = context[len(prefix_text):]
        prefix_ids = (self.tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
                      if prefix_text else [])
        answer_ids = self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
        if not answer_ids:
            raise ValueError("non-empty behavioral part produced no target tokens")
        mask_id = int(self.model.config.mask_token_id)
        max_length = int(self.model.config.max_sequence_length)
        losses = []
        with torch.inference_mode():
            for position, answer_id in enumerate(answer_ids):
                observed = prefix_ids + answer_ids[:position]
                prefix = observed[-max_length + 1:]
                query = torch.tensor([prefix + [mask_id]], device=self.input_device)
                logits = self.model.model(
                    input_ids=query, last_logits_only=True
                ).logits[0, -1].float()
                loss = -torch.log_softmax(logits, dim=-1)[int(answer_id)]
                losses.append(float(loss.item()))
        return float(sum(losses) / np.log(2)), context

    def start_neural_recording(self, recording_target, recording_type) -> None:
        if recording_target != ArtificialSubject.RecordingTarget.language_system:
            raise ValueError(f"unsupported recording target: {recording_target}")
        if recording_type not in (ArtificialSubject.RecordingType.fMRI, ArtificialSubject.RecordingType.ECoG):
            raise ValueError(f"unsupported recording type: {recording_type}")
        self._recording = (recording_target, recording_type)

    @staticmethod
    def _context_and_span(parts: list[str]) -> tuple[str, tuple[int, int]]:
        normalized = [part.strip() for part in parts]
        if not normalized or not normalized[-1]:
            raise ValueError("text parts must be non-empty")
        context = " ".join(part for part in normalized if part)
        current = normalized[-1]
        return context, (len(context) - len(current), len(context))

    def _current_representation(self, parts: list[str]) -> tuple[np.ndarray, str]:
        context, span = self._context_and_span(parts)
        key = (context, span)
        if key in self._cache:
            return self._cache[key], context
        encoded = self.tokenizer(
            context,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = encoded.to(self.input_device)
        positions = [
            index for index, (start, end) in enumerate(offsets)
            if end > start and end > span[0] and start < span[1]
        ]
        if not positions:
            raise ValueError("tokenizer did not map any token to the current text part")
        with torch.inference_mode():
            output = self.model(
                **encoded, output_hidden_states=True, output_attentions=False, return_dict=True
            )
        hidden_states = output.hidden_states
        if hidden_states is None or LAYER >= len(hidden_states):
            raise ValueError("LLaDA-8B-Base did not expose frozen layer 24")
        current_positions = torch.as_tensor(positions, device=hidden_states[LAYER].device)
        representation = (
            hidden_states[LAYER][0, current_positions, :]
            .mean(dim=0)
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        self._cache[key] = representation
        return representation, context

    def digest_text(self, text: str | Sequence[str]):
        if self._recording is None and self._behavioral_task is None:
            raise RuntimeError("start a behavioral task or neural recording before digest_text")
        texts = [str(text)] if isinstance(text, str) else [str(value) for value in text]
        if not texts:
            raise ValueError("digest_text requires at least one text part")
        parts: list[str] = []
        representations, contexts, surprisals = [], [], []
        for text_part in texts:
            parts.append(text_part)
            if self._recording is not None:
                representation, context = self._current_representation(parts)
                representations.append(representation)
            else:
                context = ""
            if self._behavioral_task is not None:
                surprisal, behavioral_context = self._part_surprisal(parts)
                surprisals.append(surprisal)
                if self._recording is None:
                    context = behavioral_context
            contexts.append(context)
        behavior = (BehavioralAssembly(
            np.asarray(surprisals, dtype=np.float32),
            coords={"stimulus": ("presentation", texts),
                    "context": ("presentation", contexts),
                    "part_number": ("presentation", np.arange(len(texts)))},
            dims=["presentation"],
        ) if surprisals else None)
        if self._recording is None:
            return {"behavior": behavior, "neural": None}
        values = np.stack(representations).astype(np.float32, copy=False)
        target, recording_type = self._recording
        width = values.shape[1]
        layer_name = f"llada.hidden_state.{LAYER}"
        units = np.arange(width)
        neural = NeuroidAssembly(
            values,
            coords={
                "stimulus": ("presentation", texts),
                "context": ("presentation", contexts),
                "part_number": ("presentation", np.arange(len(texts))),
                "layer": ("neuroid", np.repeat(layer_name, width)),
                "region": ("neuroid", np.repeat(target, width)),
                "recording_type": ("neuroid", np.repeat(recording_type, width)),
                "neuron_number_in_layer": ("neuroid", units),
                "neuroid_id": ("neuroid", np.asarray([f"{layer_name}--{unit}" for unit in units])),
            },
            dims=["presentation", "neuroid"],
        )
        return {"behavior": behavior, "neural": neural}
