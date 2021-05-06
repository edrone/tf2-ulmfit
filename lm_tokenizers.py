from transformers import AutoTokenizer, PreTrainedTokenizerFast
import sentencepiece as spm
from modelling_scripts.ulmfit_tf2 import SPMNumericalizer

""" Evaluate perplexity of a causal language model and pseudo-perplexity of a masked language model """

TOKENIZER_ARGS = {
    "auto": {},
    "polish_roberta": {"bos_token": "<bos>", "eos_token": "</s>",
                       "unk_token": "<unk>", "pad_token": "<pad>",
                       "mask_token": "<mask>"}
}

class LMTokenizerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_tokenizer(*, tokenizer_type, tokenizer_file, add_bos=False, add_eos=False, fixedlen=None):
        if tokenizer_type == 'spm':
            tok_obj = spm.SentencePieceProcessor(tokenizer_file)
            extra_opts = []
            if add_bos: extra_opts.append("bos")
            if add_eos: extra_opts.append("eos")
            tok_obj.set_encode_extra_options(":".join(extra_opts))
        elif tokenizer_type == 'spm_tf_text':
            tok_obj = SPMNumericalizer(spm_path=tokenizer_file, add_bos=add_bos, add_eos=add_eos, fixedlen=fixedlen)
        elif tokenizer_type =='polish_roberta':
            TOKENIZER_ARGS['polish_roberta']['tokenizer_file'] = tokenizer_file
            tok_obj = PreTrainedTokenizerFast(**TOKENIZER_ARGS['polish_roberta'])
        elif tokenizer_type == 'none':
            tok_obj=None
        else:
            raise ValueError(f"Unknown tokenizer type {tokenizer_type}")
        return tok_obj
