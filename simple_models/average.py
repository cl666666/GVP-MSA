import pandas as pd
import numpy as np
import sys
import argparse

sys.path.append('../GVP-MSA')
# sys.path.append('/home/chenlin/directed_evolution/gvp/GVP-MSA')

from utils import *

def delete_wt(df):
    wt_df = df[df["mutant"] == "WT"]
    df = df.drop(wt_df.index)
    df.reset_index(drop=True,inplace=True)
    return df

def handle(string):
    original = string[0]
    mutation = string[-1]
    resid = string[1:-1]
    return (original,mutation,resid)

def get_mut_detail(df):
    mutant_list = df['mutant']
    original_list = []
    mutation_list = []
    resid_list = []
    for mutant in mutant_list:
        (original,mutation,resid) = handle(mutant)
        original_list.append(original)
        mutation_list.append(mutation)
        resid_list.append(resid)
    
    df['original'] = original_list
    df['mutation'] = mutation_list
    df['resid'] = resid_list
    return df

def run_average_model(dataset_name,fold_num=5,run_method = 1):
    spearman_v_all = []
    for fold_idx in range(0,fold_num):
        datas = get_splited_data(dataset_name = dataset_name,
                                     data_split_method = 0,
                                     folder_num = fold_num,
                                     train_ratio=0.7,val_ratio=0.1,test_ratio=0.2,
                                     suffix = '')
        (train_dfs,val_dfs,test_dfs) = datas[fold_idx]
    

        train_dfs = pd.concat([train_dfs,val_dfs])
        train_dfs = delete_wt(train_dfs)

        test_dfs = delete_wt(test_dfs)
        total_average_pad = sum(train_dfs["log_fitness"])/len(train_dfs)
        train_dfs = get_mut_detail(train_dfs)
        test_dfs = get_mut_detail(test_dfs)

        train_dfs_basedon_resid = train_dfs.groupby(["resid"]).agg({'log_fitness':['min','max',"count",'mean']})
            
        # run_method ==0: direct mean
        # run_method ==1: calculating the mean before removing the highest and the lowest score
        if run_method ==0:
            test_pred = []
            out = train_dfs_basedon_resid[('log_fitness','mean')]
            for i in range(len(test_dfs)):
                line = test_dfs.iloc[i]
                if line['resid'] not in out.index:
                    test_pred.append(total_average_pad)
                else:
                    test_pred.append(out.loc[line['resid']])

        elif run_method ==1:
            train_dfs_basedon_resid['new_average'] = \
            (train_dfs_basedon_resid[("log_fitness", "mean")] * train_dfs_basedon_resid[("log_fitness", "count")]- \
            train_dfs_basedon_resid[("log_fitness", "min")]-train_dfs_basedon_resid[("log_fitness", "max")])/ \
            (train_dfs_basedon_resid[("log_fitness", "count")]-2)
            out = np.where(np.isinf(train_dfs_basedon_resid['new_average']),train_dfs_basedon_resid[("log_fitness", "mean")],train_dfs_basedon_resid['new_average'])
            train_dfs_basedon_resid['new_average'] = out

            out = train_dfs_basedon_resid['new_average']
            test_pred = []
            for i in range(len(test_dfs)):
                line = test_dfs.iloc[i]
                if line['resid'] not in out.index:
                    test_pred.append(total_average_pad)
                else:
                    test_pred.append(out.loc[line['resid']])

        
        pred_list = np.array(test_pred).squeeze()
        pred_list[np.where(np.isnan(pred_list))]=total_average_pad
        target_list = np.array(test_dfs['log_fitness'])
        spearman_v = spearman(pred_list,target_list)
        spearman_v_all.append(spearman_v)
        print('fold {}, the spearman of the average model is {}'.format(fold_idx,spearman_v))
    print('For dataset {}, the average spearman correlation of 5 fold is {}'.format(dataset_name,np.mean(np.array(spearman_v_all))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name',action='store', required=True)
    args = parser.parse_args()

    run_average_model(args.dataset_name)