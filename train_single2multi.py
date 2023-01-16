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
    test_dataset = args.test_dataset_name

    all_datasets = []
    all_datasets.extend(dataset_names)
    all_datasets.append(test_dataset)

    test_dfs = pd.read_csv('input_data/{}/{}_muti.csv'.format(test_dataset,test_dataset))
    test_dfs['dataset_name'] = test_dataset
    test_df_dict = {test_dataset:test_dfs}
    pred_ensembles = 0
    for fold_idx in range(0,args.fold_num):
        train_df_dict = {}
        val_df_dict = {}
        for dataset_name in dataset_names:
            datas = get_splited_data(dataset_name = dataset_name,
                                         data_split_method = 0,
                                         folder_num = args.fold_num,
                                         train_ratio=0.7,val_ratio=0.1,test_ratio=0.2,
                                         suffix = '_all')
            (train_dfs,val_dfs,test_dfs) = datas[fold_idx]
            train_dfs = pd.concat((train_dfs,test_dfs))

            train_df_dict[dataset_name] = train_dfs
            val_df_dict[dataset_name] = val_dfs

        datas = get_splited_data(dataset_name = test_dataset,
                                 data_split_method = 0,
                                 folder_num = args.fold_num,
                                 train_ratio=0.7,val_ratio=0.1,test_ratio=0.2,
                                 suffix = '_single')
        (train_dfs,val_dfs,test_dfs) = datas[fold_idx]
        train_dfs = pd.concat((train_dfs,test_dfs))

        oversample = math.floor(20000/len(train_dfs))
        train_dfs = pd.concat([train_dfs]*oversample)
        train_df_dict[test_dataset] = train_dfs
        val_df_dict[test_dataset] = val_dfs

        if args.classification_loss:
            data_category=True
            out_dim=3
        else: 
            data_category = False
            out_dim = 1
        gvp_msa = GVPMSA(
                output_dir=os.path.join(args.output_dir,'~'.join(dataset_names)),
                dataset_names=all_datasets,
                train_dfs_dict=train_df_dict,
                val_dfs_dict=val_df_dict,
                test_dfs_dict=test_df_dict,
                dataset_config=data_config,
                device = args.device,
                data_category=data_category,
                out_dim=out_dim,
                lr = args.lr,
                batch_size = args.batch_size,
                n_ensembles=args.n_ensembles,

                multi_train=args.multi_model,
                pdb_path_prefix = 'input_data',)
    
        gvp_msa.logger.write('training on fold {} \n'.format(fold_idx))
        
        result_dataframe = gvp_msa.train_onefold(fold_idx,epochs=args.epochs,patience=args.patience,
                       save_checkpoint=args.save_checkpoint, save_prediction=args.save_prediction)
        pred_ensembles += np.array(result_dataframe['pred'])

    dataframe = pd.DataFrame({'pred':pred_ensembles,'target':result_dataframe['target']})
    dataframe.to_csv(os.path.join(gvp_msa.output_dir,'pred_results.csv'.format(fold_idx)))

    ensembled_spearman = spearman(pred_ensembles,result_dataframe['target'])
    ensembled_ndcg = ndcg(pred_ensembles,np.array(result_dataframe['target']))
    gvp_msa.logger.write('ensemble {} fold, for test dataset {},spearman is {}, ndcg is {}\n'.format(
                   args.fold_num,test_dataset,ensembled_spearman,ensembled_ndcg))

if __name__ == "__main__":
    def str2bool(str):
        if type(str) == bool:
            return str
        else:
            return True if str.lower() == 'true' else False
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dataset_names',nargs='+', action='store', required=True)
    parser.add_argument('--test_dataset_name',action='store', required=True)

    parser.add_argument('--device',action='store', default='cuda:0', help='run on which device')

    parser.add_argument('--n_ensembles', action='store', type=int, default=1, help='number of models in ensemble')
    parser.add_argument('--fold_num', action='store', type=int, default=10, help='number of folds in ensemble')

    parser.add_argument('--esm_msa_linear_hidden', action='store', type=int, default=128, help='hidden dim of linear layer projected from MSA Transformer')
    parser.add_argument('--n_layers', action='store', type=int, default=2, help='number of GVP layers')
    parser.add_argument('--classification_loss', action='store',type=str2bool, default=False, help='penalize with classification loss')
    parser.add_argument('--multi_model', action='store',type=str2bool, default=True, help='train multi-protein, each protein have their own top parameters')

    parser.add_argument('--epochs', action='store', type=int, default=800, help='maximum epochs')
    parser.add_argument('--patience', action='store', type=int, default=100,help='patience for early stopping')
    parser.add_argument('--lr', action='store', default=5e-5,help='learning rate')
    parser.add_argument('--batch_size', action='store', type=int, default=50, help='batch size')

    parser.add_argument('--output_dir', action='store',default='results/train_single2multi', help='directory to save model, prediction, etc.')
    parser.add_argument('--save_checkpoint', action='store',type=str2bool, default=True, help='save pytorch model parameters')
    parser.add_argument('--save_prediction', action='store',type=str2bool, default=True, help='save prediction')
    args = parser.parse_args()
    main(args)
