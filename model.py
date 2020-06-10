import torch
import torch.nn as nn
from .model_class import NLU_Classify


inputs = 'am'


## Intention recognition
intent_model = NLU_Classify()
# def intent_model(inputs):
#     return 'ok'

status_model = NLU_Classify()

## NER
# entity_model = IE_NER()
def entity_model(inputs):
    return 'ok'

status = {m.__name__: m(inputs) for m in [intent_model, entity_model]}

print(status)
