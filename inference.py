import os,math,torch
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
import argparse
import numpy as np
import math
from omegaconf import OmegaConf

from utils import *
from data import *
from gvpmsa import *


def main(args):
    data_config = OmegaConf.load('data_config.yaml')
    test_dataname = args.test_dataset_name

    all_datasets = []
    all_datasets.append(test_dataname)
        
    test_dfs = pd.read_csv('input_data/{}/{}_rescaled.csv'.format(test_dataname,test_dataname))
    test_dfs['dataset_name'] = test_dataname
    test_df_dict = {test_dataname:test_dfs}
    
    
    fold_num=11
    pred_ensembles = 0
    for fold_idx in range(0,fold_num):
        train_df_dict = {}
        val_df_dict = {}
    
        if args.classification_loss:
            data_category=True
            out_dim=3
        else: 
            data_category = False
            out_dim = 1
        gvp_msa = GVPMSA(
                # output_dir=args.output_dir,
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
                pdb_path_prefix = 'input_data',
                load_model_path = os.path.join(args.load_model_path,'model_fold{}.pt'.format(fold_idx)))
        model = gvp_msa.models[0]
        loss,(pred,target),spearman_list = gvp_msa.runModel(model, None,mode='test')
        gvp_msa.logger.write('fold {}, test dataset is {}, spearman is {}\n'.format(fold_idx,test_dataname,spearman_list))
    
        result_dataframe = pd.DataFrame({'pred':pred,'target':target})
        result_dataframe.to_csv(os.path.join(args.output_dir,'pred_fold{}.csv'.format(fold_idx)))
        
        pred_ensembles += np.array(result_dataframe['pred'])
    pred_ensembles_dataframe = pd.DataFrame({'pred':pred_ensembles,'target':result_dataframe['target']})
    pred_ensembles_dataframe.to_csv(os.path.join(args.output_dir,'pred_ensemble.csv'))
        
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
    parser.add_argument('--load_model_path',action='store', required=True)
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
    
    parser.add_argument('--output_dir', action='store',default='results/inference', help='directory to save model, prediction, etc.')

    parser.add_argument('--save_checkpoint', action='store',type=str2bool, default=True, help='save pytorch model checkpoint')
    parser.add_argument('--save_prediction', action='store',type=str2bool, default=True, help='save prediction')
    args = parser.parse_args()
    main(args)


