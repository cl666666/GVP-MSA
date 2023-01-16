import os,math
import argparse
import numpy as np
import math
from omegaconf import OmegaConf

from utils import *
from data import *
from gvpmsa import *

def main(args):
    data_config = OmegaConf.load('data_config.yaml')
    dataset_name = args.train_dataset_names
    # print(args.msa_in)
    fold_num=5
    for fold_idx in range(0,fold_num):
        datas = get_splited_data(dataset_name = dataset_name,
                                     data_split_method = 0,
                                     folder_num = fold_num,
                                     train_ratio=0.7,val_ratio=0.1,test_ratio=0.2,
                                     suffix = '')
        (train_dfs,val_dfs,test_dfs) = datas[fold_idx]
    
        train_df_dict = {dataset_name:train_dfs}
        val_df_dict = {dataset_name:val_dfs}
        test_df_dict = {dataset_name:test_dfs}

        if args.classification_loss:
            data_category=True
            out_dim=3
        else: 
            data_category = False
            out_dim = 1
        gvp_msa = GVPMSA(
                output_dir=os.path.join(args.output_dir,'{}'.format(dataset_name)),
                dataset_names=[dataset_name],
                train_dfs_dict=train_df_dict,
                val_dfs_dict=val_df_dict,
                test_dfs_dict=test_df_dict,
                dataset_config=data_config,
                device = args.device,
                load_model_path = args.load_model_path,
                data_category=data_category,
                out_dim=out_dim,
                lr = args.lr,
                batch_size = args.batch_size,
                n_ensembles=args.n_ensembles,

                multi_train=args.multi_model,
                msa_in = args.msa_in,
                pdb_path_prefix = 'input_data',)
    
        gvp_msa.logger.write('training on fold {} \n'.format(fold_idx))
        
        gvp_msa.train_onefold(fold_idx,epochs=args.epochs,patience=args.patience,
                       save_checkpoint=args.save_checkpoint, save_prediction=args.save_prediction)
    
if __name__ == "__main__":
    def str2bool(str):
        if type(str) == bool:
            return str
        else:
            return True if str.lower() == 'true' else False
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dataset_names', action='store', required=True)
    parser.add_argument('--load_model_path', action='store', required=True)
    parser.add_argument('--device',action='store', default='cuda:0', help='run on which device')

    parser.add_argument('--n_ensembles', action='store', type=int, default=3, help='number of models in ensemble')

    parser.add_argument('--esm_msa_linear_hidden', action='store', type=int, default=128, help='hidden dim of linear layer projected from MSA Transformer')
    parser.add_argument('--n_layers', action='store', type=int, default=2, help='number of GVP layers')
    parser.add_argument('--classification_loss', action='store',type=str2bool, default=False, help='penalize with classification loss')
    parser.add_argument('--multi_model', action='store',type=str2bool, default=False, help='train multi-protein, each protein have their own top parameters')
    parser.add_argument('--msa_in', action='store',type=str2bool, default=True, help='add msa information into to model')

    parser.add_argument('--epochs', action='store', type=int, default=1500, help='maximum epochs')
    parser.add_argument('--patience', action='store', type=int, default=200,help='patience for early stopping')
    parser.add_argument('--lr', action='store', default=1e-5,help='learning rate')
    parser.add_argument('--batch_size', action='store', type=int, default=50, help='batch size')

    parser.add_argument('--output_dir', action='store',default='results/finetune_single_protein_random_split', help='directory to save model, prediction, etc.')
    parser.add_argument('--save_checkpoint', action='store',type=str2bool, default=False, help='save pytorch model checkpoint')
    parser.add_argument('--save_prediction', action='store',type=str2bool, default=True, help='save prediction')
    
    args = parser.parse_args()
    main(args)
