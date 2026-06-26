# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def conf_acc_format_eval(sequence: str) -> bool:
    pattern = r"\s*([0-9]+(?:\.[0-9]+)?)\s*</confidence>.*?"
    return re.match(pattern, sequence, re.DOTALL | re.MULTILINE) is not None


def compute_acc_conf_rewards(
    sequences: list[str],
    acc_reward_list: list[float],
    tokenizer,
    n_samples_per_prompt: int,
) -> tuple[list[float], list[float], list[int]]:
    conf_reward_list: list[float] = []
    mid_list: list[int] = []
    acc_reward_list = list(acc_reward_list)

    group_acc_list: list[float] = []
    for i in range(0, len(acc_reward_list), n_samples_per_prompt):
        prompt_group_rewards = acc_reward_list[i : i + n_samples_per_prompt]
        prompt_acc = sum(prompt_group_rewards) / len(prompt_group_rewards)
        group_acc_list.extend([prompt_acc] * len(prompt_group_rewards))

    for i, sequence in enumerate(sequences):
        validation_passed = conf_acc_format_eval(sequence)
        conf_reward, mid = -1.0, 0

        if validation_passed:
            match = re.search(r"\s*([0-9]+(?:\.[0-9]+)?)\s*</confidence>", sequence)
            try:
                assert match is not None
                conf_end = match.end()
                encoding = tokenizer(sequence, return_offsets_mapping=True)
                mid = encoding.char_to_token(conf_end)
                if mid is None:
                    raise ValueError("mid is None")
                confidence = float(match.group(1))
                conf_reward = -float((confidence - group_acc_list[i]) ** 2)
            except Exception:
                conf_reward = -1.0
                mid = 0
                acc_reward_list[i] = 0.0
        else:
            acc_reward_list[i] = 0.0

        conf_reward_list.append(conf_reward)
        mid_list.append(mid)

    return conf_reward_list, acc_reward_list, mid_list


@register("acc_conf")
class AccConfRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        n_samples_per_prompt: int = 1,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.n_samples_per_prompt = n_samples_per_prompt
        self.compute_score = compute_score or default_compute_score

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_extra_info: dict[str, list] = {}
        already_print_data_sources = {}

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        sequences = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = list(data.non_tensor_batch[self.reward_fn_key])
        ground_truths = [item.non_tensor_batch["reward_model"]["ground_truth"] for item in data]
        extra_infos = data.non_tensor_batch.get("extra_info", [None] * len(data))
        acc_reward_list = [
            float(self.compute_score(data_source, sequence, ground_truth, extra_info))
            for sequence, data_source, ground_truth, extra_info in zip(
                sequences, data_sources, ground_truths, extra_infos, strict=True
            )
        ]
        conf_reward_list, acc_reward_list, mid_list = compute_acc_conf_rewards(
            sequences=sequences,
            acc_reward_list=acc_reward_list,
            tokenizer=self.tokenizer,
            n_samples_per_prompt=self.n_samples_per_prompt,
        )
        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        reward_extra_info["conf_reward"] = list(conf_reward_list)
        reward_extra_info["acc_reward"] = list(acc_reward_list)
        reward_extra_info["mid"] = list(mid_list)

        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = acc_reward_list[i]

            data_source = data.non_tensor_batch[self.reward_fn_key][i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[response]", sequences[i])
                print("[acc_reward]", acc_reward_list[i])
                print("[conf_reward]", conf_reward_list[i])
                print("[mid]", mid_list[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
