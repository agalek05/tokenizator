from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from datasets import load_dataset

# Załaduj model BERT do zrozumienia kontekstu
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Załaduj zbiór danych SQuAD
dataset = load_dataset("squad")

# Przykładowy sposób przetwarzania danych z SQuAD
def process_squad_example(example):
    context = example['context']
    question = example['question']

    # Tokenizacja tekstu i pytania
    inputs = tokenizer(context, question, return_tensors='pt')

    # Uzyskanie odpowiedzi od modelu
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs['input_ids'][0, answer_start:answer_end])

    return {'context': context, 'question': question, 'answer': answer}

# Przykładowe użycie na pierwszym przykładzie ze zbioru
sample_example = dataset['train'][0]
result = process_squad_example(sample_example)
print("Example:", result)



