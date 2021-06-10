import argparse
from lm_tokenizers import LMTokenizerFactory
from corpus_feeder import LMCorpusLoader, restore_model, predict_all_arch, modify_max_seq_len
from lm_metrics import calculate_ppl

""" Evaluate perplexity of a causal language model and pseudo-perplexity of a masked language model """


def main(args):
    tokenizer_obj = LMTokenizerFactory.get_tokenizer(tokenizer_type=args['tokenizer_type'],
                                                     tokenizer_file=args['tokenizer_file'],
                                                     add_bos=args.get('add_bos'),
                                                     add_eos=args.get('add_eos'))
    corpus_loader = LMCorpusLoader(corpus_path=args['corpus_path'],
                                   batch_size=args['batch_size'],
                                   min_seq_len=args['min_seq_len'],
                                   max_seq_len=args['max_seq_len'],
                                   tokenizer_obj=tokenizer_obj)
    model = restore_model(model_type=args['model_type'], pretrained_path=args['pretrained_path'])
    if args['max_seq_len'] != model.input_shape[-1]:
        print(f"Info - original checkpoint was saved with a sequence length of {model.input_shape[-1]} - " \
              f"changing this to {args['max_seq_len']} as requested.")
        model = modify_max_seq_len(pretrained_model=model, new_max_seq_len=args['max_seq_len'])
    model.summary()
    if args['model_type'] == 'causal':
        ppl_score, num_sents = calculate_ppl(restored_model=model,
                                             corpus_loader=corpus_loader,
                                             is_pretokenized=args['is_pretokenized'],
                                             is_softmaxed=not args.get('lm_head_needs_softmaxing'))
        print(f"Perplexity = {ppl_score} (on {num_sents} sentences)")
    else:
        raise NotImplementedError(f"Unsupported model type {model_type}")

if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--corpus-path", required=True, help="Path to a corpus file. " \
                      "One line = one sentence = one training/test example, unless --as-running-text is set below")
    argz.add_argument("--model-type", required=True, choices=['causal', 'polish_roberta'], help="For causal LM - evaluate " \
                      "perplexity, for masked LM - evaluate pseudo-perplexity")
    argz.add_argument("--pretrained-path", required=True, help="Path to a directory containing a pretrained LM" \
                      "(e.g. `pytorch_model.bin`, `plwiki100.hdf5`)")
    argz.add_argument("--tokenizer-file", required=False, help="Explicit path to tokenizer.json or spm model (if present)")
    argz.add_argument("--tokenizer-type", choices=['spm', 'polish_roberta', 'none'], required=True, \
                      help="Predefined kwargs for instantiating the tokenizer loader")
    argz.add_argument("--padding-direction", choices=['pre', 'post'], default='post', \
                      help="`post` for PPPL and evaluation")
    argz.add_argument("--max-seq-len", default=768, type=int, help="Maximum sequence length")
    argz.add_argument("--min-seq-len", default=1, type=int, help="Minimum sequence length")
    argz.add_argument("--add-bos", action='store_true', help="Whether to add <s> tokens to each sentence")
    argz.add_argument("--add-eos", action='store_true', help="Whether to add </s> tokens to each sentence")
    #### MLM from transformers ####
    argz.add_argument("--mlm-from-pt", action='store_true', help="Whether the original model was trained with PyTorch")
    argz.add_argument("--mlm-tokenizer-loader", choices=['auto', 'pretrained_fast'], default='pretrained_fast', \
                      help="Which tokenizer loader class to use (AutoTokenizer or PreTrainedTokenizerFast")
    #### CAUSAL LM ####
    argz.add_argument("--is-pretokenized", action='store_true', help="Whether the corpus is pretokenized and converted to token IDs")
    argz.add_argument("--lm-head-needs-softmaxing", action='store_true', help="If set, the LM head has linear activation and scores " \
                                                                              "need to be softmaxed before evaluating PPL")
    argz.add_argument("--batch-size", type=int, default=64, help="Default batch size for predictions")

    argz = vars(argz.parse_args())

    if argz['model_type'] == 'causal':
        if argz.get('is_pretokenized') is None:
            assert argz.get('tokenizer_file') is not None, "If your corpus isn't converted to token IDs, please provide " \
                                                               "a path to the spm.model file"
        main(argz)
    else:
        raise NotImplementedError("Only causal LM metrics supported for now.")
