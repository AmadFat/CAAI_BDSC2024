import torch
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForSeq2SeqLM, pipeline, BertTokenizer


def assembled_pipeline(raw_texts):
    with torch.no_grad():
        with torch.inference_mode():
            translation_model_path = './opus-mt-zh-en'
            translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_path)
            translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_path)
            pipe = pipeline('translation', model=translation_model, tokenizer=translation_tokenizer, device='cuda:0')
            translated_texts = pipe(raw_texts)
            translated_texts = [translated_text['translation_text'] for translated_text in translated_texts]

            classification_model_path = './bert-base-personality'
            classification_tokenizer = BertTokenizer.from_pretrained(classification_model_path)
            classification_model = BertForSequenceClassification.from_pretrained(classification_model_path).cuda()

            tokenized_texts = classification_tokenizer(translated_texts, max_length=128, truncation=True, padding=True, return_tensors="pt")
            for key in tokenized_texts.keys():
                tokenized_texts[key] = tokenized_texts[key].cuda()
            outputs = classification_model(**tokenized_texts)
    predictions = outputs.logits.sigmoid().cpu().numpy()
    return predictions
