import os
import argparse
import tensorflow as tf
from ulmfit_commons import save_as_keras
from modelling_scripts.ulmfit_tf2 import ExportableULMFiT, ExportableULMFiTRagged
from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *

def main(args):
    state_dict = torch.load(open(args['pretrained_model'], 'rb'), map_location='cpu')
    state_dict = state_dict['model']
    exp_name = os.path.splitext(os.path.basename(args['pretrained_model']))[0]
    lm_num, encoder_num, outmask_num, spm_encoder_model = save_as_keras(state_dict=state_dict,
                                                                        exp_name=exp_name,
                                                                        save_path=os.path.join(args['out_path'], 'keras_weights'),
                                                                        spm_model_file=args['spm_model_file'])
    print("Exported weights successfully")
    tf.keras.backend.set_learning_phase(0)
    if args.get('fixed_seq_len') is None:
        exportable = ExportableULMFiTRagged(encoder_num, outmask_num, spm_encoder_model, state_dict['1.decoder.bias'])
        convenience_signatures={'numericalized_encoder': exportable.numericalized_encoder}
        tf.saved_model.save(exportable, os.path.join(args['out_path'], 'saved_model'), signatures=convenience_signatures)
    else:
        exportable = ExportableULMFiT(encoder_num, outmask_num, spm_encoder_model, state_dict['1.decoder.bias'])
        convenience_signatures={'numericalized_encoder': exportable.numericalized_encoder,
                                'string_encoder': exportable.string_encoder,
                                'spm_processor': exportable.string_numericalizer}
        tf.saved_model.save(exportable, os.path.join(args['out_path'], 'saved_model'), signatures=convenience_signatures)
    print("Exported SavedModel successfully. Conversion complete")

if __name__ == "__main__":
    argz = argparse.ArgumentParser(description="Loads weights from an ULMFiT .pth file trained using FastAI into a Keras model.\n" \
                                               "This script produce two output formats: weights-only and a SavedModel")
    argz.add_argument("--pretrained-model", required=True, help="Path to a pretrained FastAI model (.pth)")
    argz.add_argument("--out-path", required=True, help="Output directory where the converted TF model weights will be saved")
    argz.add_argument("--fixed-seq-len", type=int, required=False, help="(SavedModel only) Fixed sequence length. If unset, the RNN encoder will output ragged tensors.")
    argz.add_argument("--spm-model-file", required=True, help="Path to SPM model file")
    argz = vars(argz.parse_args())
    main(argz)
