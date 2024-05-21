# coding=utf-8
# Copyright 2024 The Trax Authors.
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

from transformers.models.reformer.modeling_reformer import (
    LSHSelfAttention,
    ReformerConfig,
)


class LSHAttention(LSHSelfAttention):
    '''
    https://huggingface.co/transformers/_modules/transformers/models/reformer/modeling_reformer.html
    '''
    def __init__(self, config, query, key, value):
        self.num_hash = config["num_hash"]
        reformer_config = ReformerConfig()
        reformer_config.attention_head_size = config["head_dim"]
        reformer_config.num_attention_heads = config["num_head"]
        reformer_config.attn_layers = ["lsh"]
        reformer_config.num_hashes = config["num_hash"]
        reformer_config.is_decoder = False
        reformer_config.max_position_embeddings = config["max_seq_len"]
        reformer_config.hidden_size = config["transformer_dim"]
        super().__init__(reformer_config)
        self.query_key.weight = query.weight
        self.value.weight = value.weight

    def forward(self, X, mask):
        return super().forward(hidden_states = X, attention_mask = mask).hidden_states

    def extra_repr(self):
        return f'num_hash={self.num_hash}'


