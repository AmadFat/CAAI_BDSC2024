import torch
from transformers import pipeline


def assembled_pipeline(raw_texts,
                       translation_tokenizer,
                       translation_model,
                       classification_tokenizer,
                       classification_model,):
    with torch.no_grad():
        with torch.inference_mode():
            pipe = pipeline('translation', model=translation_model, tokenizer=translation_tokenizer, device='cuda:0')
            translated_texts = pipe(raw_texts)
            translated_texts = [translated_text['translation_text'] for translated_text in translated_texts]
            tokenized_texts = classification_tokenizer(translated_texts, max_length=128, truncation=True, padding=True, return_tensors="pt")
            for key in tokenized_texts.keys():
                tokenized_texts[key] = tokenized_texts[key].cuda()
            outputs = classification_model(**tokenized_texts)
    predictions = outputs.logits.sigmoid().cpu().numpy()
    return predictions
