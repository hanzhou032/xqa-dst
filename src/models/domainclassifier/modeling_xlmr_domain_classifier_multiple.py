import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import (PreTrainedModel)
from transformers.modeling_roberta import BertLayerNorm
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
class RobertaForDomainClassifier(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaPreTrainedModel, self).__init__(config)
        self.class_labels = 2
        self.class_types = config.dst_class_types
        self.slot_list = config.dst_slot_list
        self.roberta = XLMRobertaModel(config)
        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)
        self.domain_list = ['taxi', 'restaurant', 'hotel', 'attraction', 'train']
        for domain in self.domain_list:
            self.add_module("class_"+domain, nn.Linear(config.hidden_size, self.class_labels))
        self.init_weights()

    def forward(self,
                input_ids,
                input_mask=None,
                segment_ids=None,
                position_ids=None,
                head_mask=None,
                class_label_id=None):
        outputs = self.roberta(
            input_ids,
            attention_mask=input_mask
            # token_type_ids=segment_ids,
            # position_ids=position_ids,
            # head_mask=head_mask
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
