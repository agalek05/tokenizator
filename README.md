# tokenizator
# Przetwarzanie Wiadomości i Zrozumienie Kontekstu

Ten skrypt demonstruje przykładowe implementacje dwóch podejść do przetwarzania wiadomości i zrozumienia kontekstu. Pierwsze podejście korzysta z modelu BERT do zadania Question Answering (QA) na zbiorze danych SQuAD (Stanford Question Answering Dataset). Drugie podejście używa modelu GPT-3 do uzyskania odpowiedzi na podstawie listy wiadomości.

## Podejście 1: Model BERT na SQuAD

1. **Załaduj Model BERT**

    ```python
    from transformers import BertTokenizer, BertForQuestionAnswering
    import torch
    from datasets import load_dataset

    # Załaduj model BERT do zrozumienia kontekstu
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    ```

2. **Załaduj Zbiór Danych SQuAD**

    ```python
    # Załaduj zbiór danych SQuAD
    dataset = load_dataset("squad")
    ```

3. **Przetwarzanie Przykładu ze Zbioru SQuAD**

    ```python
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
    ```

## Podejście 2: Model GPT-3

1. **Zainstaluj Bibliotekę OpenAI**

    ```bash
    pip install openai
    ```

2. **Ustawienie Klucza API OpenAI**

    ```python
    import openai

    # Ustawienie klucza API OpenAI - wymaga utworzenia konta na stronie OpenAI
    openai.api_key = 'TWÓJ_KLUCZ_API'
    ```

3. **Przetwarzanie Wiadomości za Pomocą Modelu GPT-3**

    ```python
    # Przetwarzanie przykładowych wiadomości za pomocą modelu GPT-3
    def process_messages(messages):
        # Zbuduj kompletny kontekst z listy wiadomości
        context = ' '.join(messages)

        # Użyj modelu GPT-3 do uzyskania odpowiedzi w kontekście
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=context,
            max_tokens=150
        )

        # Wyciągnij odpowiedź z wyników
        answer = response['choices'][0]['text'].strip()

        return answer

    # Przykładowe użycie
    messages = [
        "User: Cześć, co u Ciebie?",
        "Bot: Witaj! U mnie wszystko w porządku. Jak mogę Ci dzisiaj pomóc?",
        "User: Mam pytanie dotyczące zadania domowego z matematyki.",
        "Bot: Oczywiście! Jakie masz pytanie?"
    ]

    result = process_messages(messages)
    print("Odpowiedź:", result)
    ```

## Ostrzeżenie

Korzystanie z modeli językowych, takich jak BERT i GPT-3, może być związane z kosztami i wymaga uwagi w zakresie bezpieczeństwa i prywatności danych użytkowników.
