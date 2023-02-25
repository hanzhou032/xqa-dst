import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import (add_start_docstrings, add_start_docstrings_to_callable)
from transformers.modeling_bert import (BertModel, BertPreTrainedModel, BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)


@add_start_docstrings(
    """BERT Model with a classification heads for the DST task. """,
    BERT_START_DOCSTRING,
)
class BertForDomainClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForDomainClassifier, self).__init__(config)
        self.class_labels = 2
        self.class_types = config.dst_class_types
        self.slot_list = config.dst_slot_list
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)
        self.domain_list = ['taxi', 'restaurant', 'hotel', 'attraction', 'train']
        for domain in self.domain_list:
            self.add_module("class_"+domain, nn.Linear(config.hidden_size, self.class_labels))
        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(self,
                input_ids,
                input_mask=None,
                segment_ids=None,
                position_ids=None,
                head_mask=None,
                class_label_id=None):
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        total_loss = 0
        per_domain_per_example_loss = {}
        per_domain_class_logits = {}
        pooled_output_aux = pooled_output
        for index, domain in enumerate(self.domain_list):
            class_logits = self.dropout_heads(getattr(self, "class_"+domain)(pooled_output_aux))
            per_domain_class_logits[domain] = class_logits
            # If there are no labels, don't compute loss
            if class_label_id is not None:
                class_loss_fct = CrossEntropyLoss(reduction='none')
                class_loss = class_loss_fct(class_logits, class_label_id.t()[index])
                total_loss += class_loss.sum()
                per_domain_per_example_loss[domain] = class_loss

        # add hidden states and attention if they are here
        outputs = (total_loss,) + (
        per_domain_per_example_loss, per_domain_class_logits, per_domain_class_logits, per_domain_class_logits,) + outputs[2:]

        return outputs
