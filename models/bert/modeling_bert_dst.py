import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import (add_start_docstrings, add_start_docstrings_to_callable)
from transformers.modeling_bert import (BertModel, BertPreTrainedModel, BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)


@add_start_docstrings(
    """BERT Model with a classification heads for the DST task. """,
    BERT_START_DOCSTRING,
)
class BertForDST(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForDST, self).__init__(config)
        self.slot_list = config.dst_slot_list
        self.class_types = config.dst_class_types
        self.class_labels = config.dst_class_labels
        self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
        self.class_loss_ratio = config.dst_class_loss_ratio

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)
        for slot in self.slot_list:
            self.add_module("class_" + slot, nn.Linear(config.hidden_size, self.class_labels))
            self.add_module("token_" + slot, nn.Linear(config.hidden_size, 2))

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(self,
                input_ids,
                input_mask=None,
                segment_ids=None,
                position_ids=None,
                head_mask=None,
                start_pos=None,
                end_pos=None,
                inform_slot_id=None,
                refer_id=None,
                class_label_id=None,
                diag_state=None,
                aux_task_def=None):
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        total_loss = 0
        per_slot_per_example_loss = {}
        per_slot_class_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}

        for slot in self.slot_list:
            pooled_output_aux = pooled_output
            class_logits = self.dropout_heads(getattr(self, 'class_' + slot)(pooled_output_aux))

            token_logits = self.dropout_heads(getattr(self, 'token_' + slot)(sequence_output))
            start_logits, end_logits = token_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            per_slot_class_logits[slot] = class_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits

            # If there are no labels, don't compute loss
            if class_label_id is not None and start_pos is not None and end_pos is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_pos[slot].size()) > 1:
                    start_pos[slot] = start_pos[slot].squeeze(-1)
                if len(end_pos[slot].size()) > 1:
                    end_pos[slot] = end_pos[slot].squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)  # This is a single index
                start_pos[slot].clamp_(0, ignored_index)
                end_pos[slot].clamp_(0, ignored_index)

                class_loss_fct = CrossEntropyLoss(reduction='none')
                token_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)

                start_loss = token_loss_fct(start_logits, start_pos[slot])
                end_loss = token_loss_fct(end_logits, end_pos[slot])
                token_loss = (start_loss + end_loss) / 2.0

                token_is_pointable = (start_pos[slot] > 0).float()
                if not self.token_loss_for_nonpointable:
                    token_loss *= token_is_pointable

                class_loss = class_loss_fct(class_logits, class_label_id[slot])

                per_example_loss = self.class_loss_ratio * class_loss + (1 - self.class_loss_ratio) * token_loss
                total_loss += per_example_loss.sum()
                per_slot_per_example_loss[slot] = per_example_loss

        # add hidden states and attention if they are here
        outputs = (total_loss,) + (
        per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits,) + outputs[2:]

        return outputs
