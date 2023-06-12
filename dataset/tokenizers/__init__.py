from transformers import BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer, AutoTokenizer
from dataset.tokenizers.bert_tokenizer_with_dropout import BertTokenizerWithDropout


def build_tokenizer(text_encoder: str, dropout=0):
    if ('bert-base-uncased' in text_encoder) or ('bert-large-uncased' in text_encoder):
        if dropout > 0:
            tokenizer = BertTokenizerWithDropout.from_pretrained(text_encoder, dropout=dropout)
        else:
            tokenizer = BertTokenizer.from_pretrained(text_encoder)

    elif ('xlm-roberta-base' in text_encoder) or ('xlm-roberta-large' in text_encoder):
        tokenizer = XLMRobertaTokenizer.from_pretrained(text_encoder)

    elif ('roberta-base' in text_encoder) or ('roberta-large' in text_encoder):
        tokenizer = RobertaTokenizer.from_pretrained(text_encoder)

    else:
        raise NotImplementedError(f"tokenizer for {text_encoder}")

    # always use cls and sep
    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token})
    tokenizer.add_special_tokens({'eos_token': tokenizer.sep_token})

    return tokenizer