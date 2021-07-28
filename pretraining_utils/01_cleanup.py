"""
Does basic cleanup

python -m pretraining_utils.01_cleanup \
          --input-text $LM_MODELS/../datasets/books/therepublic.txt \
          --lang english

Note: several preprocessing functions in this file are not used, but we left them here
in case you have a corpus that needs more aggressive cleaning.
"""
import argparse
import logging
import re
import unicodedata

import nltk

from .polish_sentence_nltk_tokenizer import extra_abbreviations

logging.basicConfig(level=logging.INFO)


def basic_cleanup(corpus_blob, lang, sent_tokenizer):
    """ Cleans up for building SentencePiece model.
    
    corpus_blob - continuous text
    
    Returns list of sentences
    """
    corpus_blob = corpus_blob.split('\n')
    # remove wikipedia markers:
    corpus_blob = [s for s in corpus_blob if not (s.startswith("=") and s.endswith("="))]
    corpus_blob = " ".join(corpus_blob)
    sents = sent_tokenizer.tokenize(corpus_blob) # sentence-tokenize
    sents = [unicodedata.normalize('NFKC', sent) for sent in sents]
    logging.info("Delicately splitting some punctuation characters")
    sents = [re.sub(r"(\w)([,!\?])", "\\1 \\2 ", sent) for sent in sents] # that extra space after the second group is deliberate
    logging.info("Splitting the sentence-final dot")
    sents = [re.sub(r"\.$", " .", sent) for sent in sents]
    logging.info("Some more regexps...")
    sents = [re.sub(r"([\"”„\(\)])", " \\1 ", sent) for sent in sents]
    sents = [re.sub("(&quot\s*;|&amp\s;)", "", sent) for sent in sents] # remove html quirks
    sents = [re.sub("\s{2,}", " ", sent) for sent in sents] # remove extra spaces
    return sents

def recase(text, min_span=4, cap_first=False):
    """ Converts spans of min_span UPPERCASE tokens to more a likely form:

        You must INVEST AT ONCE OR LOSE an excellent OPPORTUNITY =>
        You must invest at once or lose an excellent OPPORTUNITY
    """
    ret = []
    splits = text.split()
    if len(splits) < min_span: # nothing to do
        return text
    pos = 0
    while pos < len(splits):
        upper_cnt = 0
        j = pos
        while j < len(splits):
            if splits[j].isupper():
                upper_cnt += 1
                j += 1
            else:
                break
        if upper_cnt >= min_span:
            ret.extend(list(map(lambda p: p.lower(), splits[pos:pos+upper_cnt])))
            pos += upper_cnt
        else:
            ret.append(splits[pos])
            pos += 1
    ret = " ".join(ret)
    if cap_first:
        ret = ret[0].upper() + ret[1:]
    return ret

def recase_single(text):
    """ Converts single UPPERCASE tokens added for emphasis into a more likely form:

        Downcases tokens and keeps the first letter capitalized if any surrounding token is capitalized as well
    """
    ret = []
    splits = text.split()
    i = 0
    for i, token in enumerate(splits):
        if i == 0:
            if token.isupper() and token.isalpha():
                ret.append(token[0] + token[1:].lower())
            else:
                ret.append(token)
        elif i == len(splits)-1:
            if token.isupper() and token.isalpha():
                if splits[i-1][0].isupper():
                    ret.append(token[0] + token[1:].lower())
                else:
                    ret.append(token.lower())
            else:
                ret.append(token)
        else:
            if token.isupper() and token.isalpha():
                if splits[i-1][0].isupper() or splits[i+1][0].isupper():
                    ret.append(token[0] + token[1:].lower())
                else:
                    ret.append(token.lower())
            else:
                ret.append(token)
    ret = " ".join(ret)
    return ret

def attempt_split(text, hunspell_obj, min_len=20):
    """ Attempts to split words that were accidentally merged, e.g.
        'narzędziaklucz' => 'narzędzia klucz'
    """
    ret = []
    splits = text.split()
    for fake_token in splits:
        if len(fake_token) < min_len:
            ret.append(fake_token)
        else:
            splitted_candidate = hunspell_obj.suggest(fake_token.strip(',.-+!?"')) # punctuation goes to hell here...
            if splitted_candidate == []:
                ret.append(fake_token)
            else:
                ret.append(splitted_candidate[0])
    return " ".join(ret)

def strip_hashtags(text):
    return re.sub('#.*?\b', ' ', text)

def attempt_split_uppercase(text, min_upper=4, recase=True, last_upper_goes_to='right'):
    """ Splits phrases that were accidentally merged during crawling based on the letter case:

         Ex) UWAGISmak i konsystencja => UWAGI Smak i konsystencja   (recase=False, last_upper_goes_to='right')
                                      => UWAGIS mak i konsystencja   (recase=False, last_upper_goes_to='left')
                                      => Uwagi Smak i konsystencja   (recase=True, last_upper_goes_to='right')
                                      => Uwagis mak i konsystencja   (recase=True, last_upper_goes_to='left')
    """
    regexp = re.compile("([A-ZĄĘŚĆŃŹŻÓŁ]{" + str(min_upper) + ",})([a-ząęśćńźżół]{2,})")
    TO_LEFT = lambda p: p.group(1) + " " + p.group(2)
    TO_RIGHT = lambda p: p.group(1)[:-1] + " " + p.group(1)[-1]+p.group(2)
    TO_LEFT_RECASE = lambda p: p.group(1)[0]+p.group(1).lower()[1:] + " " + p.group(2)
    TO_RIGHT_RECASE = lambda p: p.group(1)[0]+p.group(1).lower()[1:-1] + " " + p.group(1)[-1]+p.group(2)
    if last_upper_goes_to == 'right':
        tfm = TO_RIGHT_RECASE if recase else TO_RIGHT
    elif last_upper_goes_to == 'left':
        tfm = TO_LEFT_RECASE if recase else TO_LEFT
    else:
        raise ValueError(f"Unknown operation on the boundary uppercase letter: {last_upper_goes_to}")
    return re.sub(regexp, tfm, text)


def attempt_split_hill(text, min_lower_left=4, min_lower_right=3):
    """ Splits phrases that were accidentally merged during crawling based on the letter case:

         Ex) zapasMydła => zapasMydła
    """
    regexp = re.compile("([a-ząęśćńźżół]{" + str(min_lower_left) + \
                        ",})([A-ZĄĘŚĆŃŹŻÓŁ])([a-ząęśćńźżół]{" + str(min_lower_right) + ",})")
    tfm = lambda p: p.group(1) + " " + p.group(2) + p.group(3)
    is_match = re.search(regexp, text)
    if is_match:
        print(f"{is_match}")
    return re.sub(regexp, tfm, text)


def main(args):
    logging.info(f"Reading input file {args['input_text']}")
    with open(args['input_text'], 'r', encoding='utf-8') as f:
        corpus_blob = f.read()
    logging.info(f"Cleaning up the input file...")
    sent_tokenizer = nltk.data.load(f'tokenizers/punkt/{args["lang"]}.pickle')
    if args['lang'] == 'polish':
        sent_tokenizer._params.abbrev_types.update(extra_abbreviations)
    sents = basic_cleanup(corpus_blob, args['lang'], sent_tokenizer)
    if args.get('uncased') is True:
        logging.info(f"Downcasing...")
        sents = [sent.lower() for sent in sents]
        sents = [re.sub("(&quot\s*;|&amp\s;)", "", sent) for sent in sents]
    with open(f"{args['out_path']}", "w", encoding="utf-8") as f:
        for sent in sents: f.write(sent + "\n")
    logging.info("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs sentence tokenization on an input text file "
                                                 "and saves the result as a text file with one sentence in each line")
    parser.add_argument("--lang", required=True, help="NLTK language name for sentence tokenization.")
    parser.add_argument("--input-text", required=True, help="Path to a raw text corpus. One big single file.")
    parser.add_argument("--uncased", required=False, action='store_true', help="Downcase everything")
    parser.add_argument("--out-path", required=True, help="Path where the results will be saved.")
    argz = parser.parse_args()
    main(vars(argz))
