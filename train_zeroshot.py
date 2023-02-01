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
    dataset_names = args.train_dataset_names
    test_dataname = args.test_dataset_name

    all_datasets = []
    all_datasets.extend(dataset_names)
    all_datasets.append(test_dataname)
        
    test_dfs = pd.read_csv('./input_data/{}/{}_rescaled.csv'.format(test_dataname,test_dataname))
    test_dfs['dataset_name'] = test_dataname
    test_df_dict = {test_dataname:test_dfs}
    
    
    fold_num=len(dataset_names)
    pred_ensembles = 0
    for fold_idx in range(0,fold_num):
        train_df_dict = {}
        val_df_dict = {}
        val_dataset_name = dataset_names[fold_idx]
        for dataset_name in dataset_names:
            dfs = pd.read_csv('./input_data/{}/{}_rescaled.csv'.format(dataset_name,dataset_name))
            dfs['dataset_name'] = dataset_name
            if dataset_name != val_dataset_name:
                train_df_dict[dataset_name] = dfs
            else:
                val_df_dict[dataset_name] = dfs
    
        if args.classification_loss:
            data_category=True
            out_dim=3
        else: 
            data_category = False
            out_dim = 1
        gvp_msa = GVPMSA(
                output_dir=args.output_dir,
                dataset_names=all_datasets,
                train_dfs_dict=train_df_dict,
                val_dfs_dict=val_df_dict,
                test_dfs_dict=test_df_dict,
                dataset_config=data_config,
                device = args.device,
                data_category=data_category,
                out_dim=out_dim,
                batch_size = args.batch_size,
                n_ensembles=args.n_ensembles,
                lr = args.lr,
                multi_train=args.multi_model,esm_msa_linear_hidden = args.esm_msa_linear_hidden,
                pdb_path_prefix = './input_data',)
        
        gvp_msa.logger.write('fold {}, train&val dataset is {},\nval dataset is {},\ntest dataset is {}\n'.format(
                       fold_idx,','.join(dataset_names),val_dataset_name,test_dataname))
    
    
        result_dataframe = gvp_msa.train_onefold(fold_idx,epochs=args.epochs,patience=args.patience,
                       save_checkpoint=args.save_checkpoint, save_prediction=args.save_prediction)
        pred_ensembles += np.array(result_dataframe['pred'])

    ensembled_spearman = spearman(pred_ensembles,result_dataframe['target'])
    ensembled_ndcg = ndcg(pred_ensembles,np.array(result_dataframe['target']))
    gvp_msa.logger.write('ensemble {} fold, for test dataset {},spearman is {}, ndcg is {}\n'.format(
                   fold_num,test_dataname,ensembled_spearman,ensembled_ndcg))


if __name__ == "__main__":
    def str2bool(str):
        if type(str) == bool:
            return str
        else:
            return True if str.lower() == 'true' else False
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dataset_names',nargs='+',action='store', required=True)
    parser.add_argument('--test_dataset_name',action='store', required=True)

    parser.add_argument('--device',action='store', default='cuda:1', help='run on which device')

    parser.add_argument('--n_ensembles', action='store', type=int, default=1, help='number of models in ensemble')

    parser.add_argument('--esm_msa_linear_hidden', action='store', type=int, default=128, help='hidden dim of linear layer projected from MSA Transformer')
    parser.add_argument('--n_layers', action='store', type=int, default=2, help='number of GVP layers')
    parser.add_argument('--classification_loss', action='store',type=str2bool, default=True, help='penalize with classification loss')
    parser.add_argument('--multi_model', action='store',type=str2bool, default=False, help='train multi-protein, each protein have their own top parameters')

    parser.add_argument('--epochs', action='store', type=int, default=80, help='total epochs')
    parser.add_argument('--patience', action='store', type=int, default=15,help='patience for early stopping')
    parser.add_argument('--batch_size', action='store', type=int, default=50, help='batch size')
    parser.add_argument('--lr', action='store', default=5e-5,help='learning rate')
    
    parser.add_argument('--output_dir', action='store',default='results/zeroshot', help='directory to save model, prediction, etc.')

    parser.add_argument('--save_checkpoint', action='store',type=str2bool, default=False, help='save pytorch model checkpoint')
    parser.add_argument('--save_prediction', action='store',type=str2bool, default=True, help='save prediction')
    args = parser.parse_args()
    main(args)
