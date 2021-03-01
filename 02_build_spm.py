import argparse
import logging
import sentencepiece as spm

"""
Trains SPM model

python ./02_build_spm.py --pretokenized-text $LM_MODELS/../datasets/books/therepublic_pretokenized.txt --vocab-size 5000 --model-prefix plato

"""

logging.basicConfig(level=logging.INFO)

def main(args):
    logging.info(f"Reading input file {args['pretokenized_text']}")
    model_prefix = f"{args['model_prefix']}-sp{args['vocab_size']//1000}k"
    logging.info(f"Fitting the SPM tokenizer - will save to {model_prefix}")
    spm.SentencePieceTrainer.train(input=args['pretokenized_text'], vocab_size=args['vocab_size'], \
                                   unk_id=0, pad_id=1, bos_id=2, eos_id=3, model_prefix=model_prefix, \
                                   model_type='bpe', character_coverage=1.0, num_threads=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretokenized-text", required=True, help="Path to a raw text corpus. One big single file.")
    parser.add_argument("--vocab-size", required=True, type=int, help="How big should be the SPM vocab")
    parser.add_argument("--model-prefix", required=False, default="customspm", help="SPM model prefix")
    argz = parser.parse_args()
    main(vars(argz))
