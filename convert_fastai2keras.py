import argparse
import os
import shutil

import tensorflow as tf

from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *
from fastai_lm_utils import save_as_keras
from ulmfit_tf2 import ExportableULMFiT, ExportableULMFiTRagged, STLRSchedule


def main(args):
    os.makedirs(args['out_path'], exist_ok=True)
    state_dict = torch.load(open(args['pretrained_model'], 'rb'), map_location='cpu')
    state_dict = state_dict['model']
    exp_name = os.path.splitext(os.path.basename(args['pretrained_model']))[0]
    lm_num, encoder_num, outmask_num, spm_encoder_model = save_as_keras(state_dict=state_dict,
                                                                        exp_name=exp_name,
                                                                        save_path=os.path.join(args['out_path'], 'keras_weights'),
                                                                        spm_model_file=args['spm_model_file'],
                                                                        fixed_seq_len=args.get('fixed_seq_len'))
    print("Exported weights successfully")
    tf.keras.backend.set_learning_phase(0)
    if args.get('fixed_seq_len') is None:
        exportable = ExportableULMFiTRagged(encoder_num, outmask_num, spm_encoder_model, state_dict['1.decoder.bias'], STLRSchedule)
        convenience_signatures = {'numericalized_encoder': exportable.numericalized_encoder,
                                  'string_encoder': exportable.string_encoder,
                                  'spm_processor': exportable.string_numericalizer}
        tf.saved_model.save(exportable, os.path.join(args['out_path'], 'saved_model'), signatures=convenience_signatures)
    else:
        exportable = ExportableULMFiT(encoder_num, outmask_num, spm_encoder_model, state_dict['1.decoder.bias'])
        convenience_signatures={'numericalized_encoder': exportable.numericalized_encoder,
                                'string_encoder': exportable.string_encoder,
                                'spm_processor': exportable.string_numericalizer}
        tf.saved_model.save(exportable, os.path.join(args['out_path'], 'saved_model'), signatures=convenience_signatures)
    print("Exported SavedModel successfully.")
    os.makedirs(os.path.join(args['out_path'], 'fastai_model'), exist_ok=True)
    shutil.copy2(args['pretrained_model'], os.path.join(args['out_path'], 'fastai_model/'))
    print("FastAI model copied.")
    os.makedirs(os.path.join(args['out_path'], 'spm_model'), exist_ok=True)
    shutil.copy2(args['spm_model_file'], os.path.join(args['out_path'], 'spm_model/'))
    shutil.copy2(args['spm_model_file'].replace(".model", ".vocab"), os.path.join(args['out_path'], 'spm_model/'))
    print("SPM model copied. Conversion complete.")

if __name__ == "__main__":
    argz = argparse.ArgumentParser(description="Loads weights from an ULMFiT .pth file trained using FastAI into a Keras model.\n" \
                                               "This script will produce four subdirectories: 1) Keras weights, 2) SavedModel, 3) SPM model, 4) FastAI model")
    argz.add_argument("--pretrained-model", required=True, help="Path to a pretrained FastAI model (.pth)")
    argz.add_argument("--out-path", required=True, help="Output directory where the converted TF model weights will be saved")
    argz.add_argument("--fixed-seq-len", type=int, required=False, help="(SavedModel only) Fixed sequence length. If unset, the RNN encoder will output ragged tensors.")
    argz.add_argument("--spm-model-file", required=True, help="Path to SPM model file")
    argz = vars(argz.parse_args())
    main(argz)
