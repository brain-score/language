"""Fixed clean-neural and one-mask behavioral Brain-Score subject for Dream."""

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


MODEL_ID = "Dream-org/Dream-v0-Base-7B"
MODEL_REVISION = "6572adb5535263e4d1a337b56942ba48b6dee2a9"
LAYER = 21
BEHAVIOR_CONTEXT_TOKENS = 4095


class DreamV0BaseSubject(ArtificialSubject):
    def __init__(self, *, model_path: str | Path | None = None) -> None:
        resolved = model_path or os.environ.get("DREAM_MODEL_PATH") or MODEL_ID
        kwargs = {"trust_remote_code": True}
        if isinstance(resolved, Path) or Path(str(resolved)).is_dir():
            kwargs["local_files_only"] = True
        else:
            kwargs["revision"] = MODEL_REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(str(resolved), **kwargs)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(str(resolved), torch_dtype=dtype, **kwargs)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        self._recording = None
        self._behavioral_task = None

    def identifier(self) -> str:
        return "dream-v0-base-7b"

    def start_behavioral_task(self, task: ArtificialSubject.Task) -> None:
        if task != ArtificialSubject.Task.reading_times:
            raise NotImplementedError(f"unsupported behavioral task: {task}")
        self._behavioral_task = task

    def start_neural_recording(self, recording_target, recording_type) -> None:
        if recording_target != ArtificialSubject.RecordingTarget.language_system:
            raise ValueError(f"unsupported recording target: {recording_target}")
        if recording_type not in (ArtificialSubject.RecordingType.fMRI, ArtificialSubject.RecordingType.ECoG):
            raise ValueError(f"unsupported recording type: {recording_type}")
        self._recording = (recording_target, recording_type)

    def _ids_for_part(self, parts: list[str], *, behavior: bool):
        cleaned = [str(part).strip() for part in parts]
        if not cleaned:
            raise ValueError("at least one text part is required")
        context = " ".join(part for part in cleaned if part)
        if behavior:
            context = re.sub(r"\s+([.,!?;:])", r"\1", context)
        current = re.sub(r"\s+([.,!?;:])", r"\1", cleaned[-1]) if behavior else cleaned[-1]
        if not current:
            return [], [], context
        prefix_text = context[:-len(current)].rstrip()
        continuation = context[len(prefix_text):]
        prefix_ids = (self.tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
                      if prefix_text else [])
        current_ids = self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
        if not current_ids:
            raise ValueError("non-empty current part produced no tokenizer tokens")
        return prefix_ids, current_ids, context

    def _part_surprisal(self, parts: list[str]):
        prefix_ids, current_ids, context = self._ids_for_part(parts, behavior=True)
        if not current_ids:
            return 0.0, context
        mask_id = int(self.model.config.mask_token_id)
        nats = 0.0
        with torch.inference_mode():
            for index, target_id in enumerate(current_ids):
                observed = (prefix_ids + current_ids[:index])[-BEHAVIOR_CONTEXT_TOKENS:]
                query = torch.tensor([observed + [mask_id]], device=self.input_device)
                logits = self.model(
                    input_ids=query, use_cache=False, num_logits_to_keep=1
                ).logits[0, -1].float()
                nats -= float(torch.log_softmax(logits, dim=-1)[int(target_id)].item())
        return nats / np.log(2), context

    def _part_neural(self, parts: list[str]):
        prefix_ids, current_ids, context = self._ids_for_part(parts, behavior=False)
        if not current_ids:
            raise ValueError("neural text part contains no tokenizer tokens")
        all_ids = prefix_ids + current_ids
        if len(all_ids) > int(self.model.config.max_position_embeddings):
            raise ValueError("neural input exceeds Dream's declared context window")
        query = torch.tensor([all_ids], device=self.input_device)
        with torch.inference_mode():
            output = self.model(input_ids=query, use_cache=False,
                                output_hidden_states=True, return_dict=True)
        if output.hidden_states is None or len(output.hidden_states) <= LAYER:
            raise ValueError("Dream did not expose frozen layer 21")
        hidden = output.hidden_states[LAYER][0, len(prefix_ids):, :]
        return hidden.float().mean(dim=0).cpu().numpy().astype(np.float32, copy=False), context

    def digest_text(self, text: str | Sequence[str]):
        if self._recording is None and self._behavioral_task is None:
            raise RuntimeError("start a behavioral task or neural recording before digest_text")
        texts = [str(text)] if isinstance(text, str) else [str(value) for value in text]
        if not texts:
            raise ValueError("digest_text needs at least one text part")
        parts, contexts, behavioral, neural_values = [], [], [], []
        for part in texts:
            parts.append(part)
            if self._recording is not None:
                value, neural_context = self._part_neural(parts)
                neural_values.append(value)
            if self._behavioral_task is not None:
                surprisal, behavioral_context = self._part_surprisal(parts)
                behavioral.append(surprisal)
            contexts.append(neural_context if self._recording is not None else behavioral_context)
        behavior = (BehavioralAssembly(np.asarray(behavioral, dtype=np.float32),
                    coords={"stimulus": ("presentation", texts),
                            "context": ("presentation", contexts),
                            "part_number": ("presentation", np.arange(len(texts)))},
                    dims=["presentation"]) if behavioral else None)
        if self._recording is None:
            return {"behavior": behavior, "neural": None}
        values = np.stack(neural_values).astype(np.float32, copy=False)
        target, recording_type = self._recording
        width = values.shape[1]
        layer_name = f"dream.hidden_state.{LAYER}"
        units = np.arange(width)
        neural = NeuroidAssembly(values, coords={
            "stimulus": ("presentation", texts),
            "context": ("presentation", contexts),
            "part_number": ("presentation", np.arange(len(texts))),
            "layer": ("neuroid", np.repeat(layer_name, width)),
            "region": ("neuroid", np.repeat(target, width)),
            "recording_type": ("neuroid", np.repeat(recording_type, width)),
            "neuron_number_in_layer": ("neuroid", units),
            "neuroid_id": ("neuroid", np.asarray([f"{layer_name}--{unit}" for unit in units])),
        }, dims=["presentation", "neuroid"])
        return {"behavior": behavior, "neural": neural}
