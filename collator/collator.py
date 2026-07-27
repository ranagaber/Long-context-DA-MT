from dataclasses import dataclass
from typing import List, Dict, Any
import torch


@dataclass
class DynamicCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_labels = [f["labels"] for f in features]

        max_length = max(len(ids) for ids in batch_input_ids)

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(batch_input_ids, batch_labels):
            pad_len = max_length - len(input_ids)

            padded_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * pad_len
            )

            padded_labels.append(
                labels + [-100] * pad_len
            )

            attention_masks.append(
                [1] * len(input_ids) + [0] * pad_len
            )

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }