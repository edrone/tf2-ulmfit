import argparse
import unicodedata
import logging
import nltk
import re

logging.basicConfig(level=logging.INFO)

"""
Does basic cleanup

python ./01_cleanup.py \
       --input-text $LM_MODELS/../datasets/books/therepublic.txt \
       --lang english

"""


def basic_cleanup(corpus_blob, lang):
    """ Cleans up for building SentencePiece model.
    
    corpus_blob - continuous text
    
    Returns list of sentences
    """
    corpus_blob = corpus_blob.split('\n')
    corpus_blob = [s for s in corpus_blob if not (s.startswith("=") and s.endswith("="))]
    corpus_blob = " ".join(corpus_blob)
    sents = nltk.sent_tokenize(corpus_blob, language=lang) # sentence-tokenize
    sents = [re.sub("(&quot\s*;|&amp\s;)", "", sent) for sent in sents] # remove html quirks
    sents = [unicodedata.normalize('NFKC', sent) for sent in sents]
    return sents

def main(args):
    logging.info(f"Reading input file {args['input_text']}")
    with open(args['input_text'], 'r', encoding='utf-8') as f:
        corpus_blob = f.read()
    logging.info(f"Cleaning up the input file...")
    sents = basic_cleanup(corpus_blob, args['lang'])
    if args.get('uncased') is True:
        logging.info(f"Downcasing...")
        sents = [sent.lower() for sent in sents]
        sents = [re.sub("(&quot\s*;|&amp\s;)", "", sent) for sent in sents]
    with open(f"{args['input_text']}_pretokenized.txt", "w", encoding="utf-8") as f:
        for sent in sents: f.write(sent + "\n")
    logging.info("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="NLTK language name for sentence tokenization.")
    parser.add_argument("--input-text", required=True, help="Path to a raw text corpus. One big single file.")
    parser.add_argument("--uncased", required=False, action='store_true', help="Downcase everything")
    argz = parser.parse_args()
    main(vars(argz))
