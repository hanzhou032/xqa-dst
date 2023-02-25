# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import (add_start_docstrings, add_start_docstrings_to_callable)
from transformers.modeling_utils import (PreTrainedModel)
from transformers.modeling_roberta import (RobertaModel, RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
                                           ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING, BertLayerNorm)
from transformers.modeling_xlm_roberta import (XLMRobertaModel, XLMRobertaConfig, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, XLM_ROBERTA_START_DOCSTRING)

class RobertaPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@add_start_docstrings(
    """RoBERTa Model with classification heads for the DST task. """,
    XLM_ROBERTA_START_DOCSTRING,
)
class RobertaForDST(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForDST, self).__init__(config)
        self.slot_list = config.dst_slot_list
        self.class_types = config.dst_class_types
        self.class_labels = config.dst_class_labels
        self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
        self.class_loss_ratio = config.dst_class_loss_ratio

        self.roberta = XLMRobertaModel(config)
        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)

        self.add_module("class_common", nn.Linear(config.hidden_size, self.class_labels))
        self.add_module("token_common", nn.Linear(config.hidden_size, 2))
        self.add_module("domain_common", nn.Linear(config.hidden_size, 2))

        self.init_weights()

    # @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(self,
                input_ids,
                input_mask=None,
                segment_ids=None,
                position_ids=None,
                head_mask=None,
                start_pos=None,
                end_pos=None,
                class_label_id=None,
                domain_label_id=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=input_mask
            # token_type_ids=segment_ids,
            # position_ids=position_ids,
            # head_mask=head_mask
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        total_loss = 0
        per_slot_per_example_loss = {}

        pooled_output_aux = pooled_output
        class_logits = self.dropout_heads(getattr(self, "class_common")(pooled_output_aux))
        domain_logits = self.dropout_heads(getattr(self, "domain_common")(pooled_output_aux))
        token_logits = self.dropout_heads(getattr(self, "token_common")(sequence_output))
        start_logits, end_logits = token_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        per_slot_class_logits = class_logits
        per_slot_start_logits = start_logits
        per_slot_end_logits = end_logits

        # If there are no labels, don't compute loss
        if class_label_id is not None and start_pos is not None and end_pos is not None and domain_label_id is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_pos.size()) > 1:
                start_pos = start_pos.squeeze(-1)
            if len(end_pos.size()) > 1:
                end_pos = end_pos.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)  # This is a single index
            start_pos.clamp_(0, ignored_index)
            end_pos.clamp_(0, ignored_index)

            class_loss_fct = CrossEntropyLoss(reduction='none')
            domain_loss_fct = CrossEntropyLoss(reduction='none')
            token_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)

            start_loss = token_loss_fct(start_logits, start_pos)
            end_loss = token_loss_fct(end_logits, end_pos)
            token_loss = (start_loss + end_loss) / 2.0

            token_is_pointable = (start_pos > 0).float()
            if not self.token_loss_for_nonpointable:
                token_loss *= token_is_pointable
            domain_is_pointable = (domain_label_id > 0).float()
            # token_loss *= domain_is_pointable

            class_loss = class_loss_fct(class_logits, class_label_id)
            # class_loss *= domain_is_pointable
            domain_loss = class_loss_fct(domain_logits, domain_label_id)
            per_example_loss = (self.class_loss_ratio * class_loss + (1 - self.class_loss_ratio) * token_loss + domain_loss)/2
            total_loss += per_example_loss.sum()
            per_slot_per_example_loss = per_example_loss

        # add hidden states and attention if they are here
        outputs = (total_loss,) + (per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, domain_logits,) + outputs[2:]

        return outputs
