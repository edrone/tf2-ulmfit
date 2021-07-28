"""
Encode corpus using a pretrained SPM model

python -m pretraining_utils.03_encode_spm \
          --corpus-path plato_therepublic.txt \
          --model-path ./plato-sp5k.model \
          --spm-extra-options bos:eos \
          --output-format id \
          --save-path ./encoded_ids.txt \
          --save-stats
"""
import argparse
import logging
import os
import statistics

import sentencepiece as spm

logging.basicConfig(level=logging.INFO)

def main(args):
    logging.info(f"Loading SPM model from {args['model_path']}")
    spmproc = spm.SentencePieceProcessor(args['model_path'])
    spmproc.set_encode_extra_options(args['spm_extra_options'])
    spmproc_encoder_fn = spmproc.encode_as_ids if args['output_format'] == 'id' \
                                               else spmproc.encode_as_pieces
    cnt = 0
    sent_lengths = []
    fsize = os.path.getsize(args['corpus_path'])
    f_src = open(args['corpus_path'], 'r', encoding='utf-8')
    f_dst = open(args['save_path'], 'w', encoding='utf-8')
    for input_sentence in f_src:
        if cnt % 10000 == 0: logging.info(f"Tokenizing sentence {cnt}")
        if args.get('lower'): input_sentence = input_sentence.lower()
        line = spmproc_encoder_fn(input_sentence.strip())
        sent_lengths.append(len(line))
        f_dst.write(" ".join(map(str, line)) + "\n")
        cnt += 1
    f_dst.close()
    f_src.close()
    sent_lengths.sort()
    q1 = sent_lengths[len(sent_lengths) // 4]
    q2 = sent_lengths[len(sent_lengths) // 2]
    q3 = sent_lengths[int(len(sent_lengths) * (3/4))]
    avg_tokens = sum(sent_lengths) // len(sent_lengths)
    std_dev = round(statistics.stdev(sent_lengths), 2)
    summary_text = "======= SUMMARY STATISTICS =======\n"
    summary_text += f"  * Number of sentences seen in the corpus: {len(sent_lengths)}\n"
    summary_text += f"  * Total number of tokens: {sum(sent_lengths)}\n"
    summary_text += f"  * Number of tokens by quartile: Q1 = {q1}, Q2/median = {q2}, Q3 = {q3}, longest = {sent_lengths[-1]}\n"
    summary_text += f"  * Mean number of tokens in each sentence: {avg_tokens} \n"
    summary_text += f"  * Standard deviation: {std_dev}, mu + 1s = {avg_tokens+std_dev}, mu + 2s = {avg_tokens+(2*std_dev)}, mu + 3s = {avg_tokens+(3*std_dev)}\n"
    summary_text += "=================================="
    print(summary_text)
    if args.get('save_stats') is True:
        with open(args['save_path'] + '.stats', 'w', encoding='utf-8') as f:
            f.write(summary_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", required=True, help="Path to a raw text corpus (cleaned up and preprocessed). One line = one sentence.")
    parser.add_argument("--model-path", required=True, help="Path to .model file")
    parser.add_argument("--spm-extra-options", default="bos:eos", help="SPM extra options")
    parser.add_argument("--lower", action='store_true', help="Downcase (if not done already) before encoding")
    parser.add_argument("--output-format", choices=['piece', 'id'], default="id", help="SPM output format")
    parser.add_argument("--save-path", required=True, help="Output path for an encoded corpus")
    parser.add_argument("--save-stats", action='store_true', help="Save corpus statistics")
    argz = parser.parse_args()
    main(vars(argz))
