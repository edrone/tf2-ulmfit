import argparse
import json
import tensorflow as tf
from awd_lstm_tensorflow2.ulmfit_tf2_heads import SPMNumericalizer

def pretty_print_tagged_sequences(subword_pieces, labels):
    for i in range(len(subword_pieces)):
        l = list(zip(subword_pieces[i], labels[i]))
        print(l)

def mk_labels(ls_json):
    labels_set = set()
    for document in ls_json:
        all_tagged = document['label']
        for tagged in all_tagged:
            labels_set.update(tagged['labels'])
    label_index = {label:index for index,label in enumerate(sorted(labels_set))}
    # index_label = {index:label for index,label in enumerate(sorted(labels_set))}
    return label_index

def label_studio_to_tagged_subwords(*, spm_args, label_studio_min_json):
    spm_layer = SPMNumericalizer(name="SPM_layer",
                                 spm_path=spm_args['spm_model_file'],
                                 add_bos=spm_args.get('add_bos') or False,
                                 add_eos=spm_args.get('add_eos') or False,
                                 lumped_sents_separator="")
    spmproc = spm_layer.spmproc
    ls_json = json.load(open(label_studio_min_json, 'r', encoding='utf-8'))
    # labels_map = predefined_labels or mk_labels(ls_json)
    # print(labels_map)

    # tokenize with offsets
    nl_texts = [document['text'] for document in ls_json]
    spans = [document['label'] for document in ls_json]
    print(f"First 10 texts:")
    print(nl_texts[0:10])

    # map spans to subwords
    tokenized = []
    tokenized_pieces = []
    tokenized_labels = []
    for doc_id in range(len(nl_texts)):
        # Tensorflow's tokenize_with_offsets is broken with SentencePiece
        #token_offsets = list(zip(begins[i].numpy().tolist(), ends[i].numpy().tolist()))
        #pieces = [t.decode(encoding='utf-8') for t in spmproc.id_to_string(piece_ids[i]).numpy().tolist()]
        curr_tokens = []
        curr_pieces = []
        curr_entities = []
        i = 0
        for span in spans[doc_id]:
            j = entity_beg = span['start']
            entity_end = span['end']
            label_class = span['labels'][0] # assume labels don't overlap
            
            # tokenize everything before the label span
            res = spmproc.tokenize(nl_texts[doc_id][i:j]).numpy().tolist()
            curr_tokens.extend(res)
            curr_entities.extend(['O']*len(res))
            
            # inside the label span
            res = spmproc.tokenize(nl_texts[doc_id][j:entity_end]).numpy().tolist()
            curr_tokens.extend(res)
            curr_entities.extend([label_class]*len(res))

        # from the last label to EOS
        res = spmproc.tokenize(nl_texts[doc_id][entity_end:]).numpy().tolist()
        curr_tokens.extend(res)
        curr_entities.extend(['O']*len(res))
        curr_pieces = [t.decode(encoding='utf-8') for t in spmproc.id_to_string(curr_tokens).numpy().tolist()]
        tokenized.append(curr_tokens)
        tokenized_pieces.append(curr_pieces)
        tokenized_labels.append(curr_entities)
    return tokenized, tokenized_pieces, tokenized_labels

def main(args):
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': False, 'add_eos': False}
    token_ids, token_pieces, token_labels = label_studio_to_tagged_subwords(spm_args=spm_args,
                                            label_studio_min_json=args['label_studio_min_json'])
    pretty_print_tagged_sequences(token_pieces, token_labels)


if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument('--label-studio-min-json', required=False)
    argz.add_argument('--spm-model-file', required=False)
    args = vars(argz.parse_args())
    main(args)
    
