import numpy as np
import pandas as pd
import argparse

from scipy.stats import spearmanr


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
def runModel(train_df, test_df):

    train_df = delete_wt(train_df)
    train_df = get_mut_detail(train_df)
    test_df = delete_wt(test_df)

    single_fitness_dict = dict(zip(train_df['mutant'],train_df['log_fitness']))
    double_mutants = list(test_df['mutant'])
    if ',' in double_mutants[0]:
        sep = ','
    elif '-' in double_mutants[0]:
        sep = '-'
    else:
        raise ValueError('this is not double mut')
    nosingle_list = []
    for i in double_mutants:
        for single in i.split(sep):
            if single not in single_fitness_dict:
                nosingle_list.append(single)
    nosingle_list = list(set(nosingle_list))
    complete_ratio = len(train_df)/(len(train_df)+len(nosingle_list))
    print('the number of double mutants taht have single value is {}, no single is {}, complete_ratio: {}'.format(
        len(train_df),len(nosingle_list),complete_ratio))

    train_dfs_basedon_resid = train_df.groupby(["resid"]).agg({'log_fitness':['min','max',"count",'mean']})
    test_pred = []
    out = train_dfs_basedon_resid[('log_fitness','mean')]
    total_average_pad = np.mean(out)
    for mutant in nosingle_list:
        _,_,idex = handle(mutant)
        if idex not in out.index:
            test_pred.append(total_average_pad)
        else:
            test_pred.append(out.loc[idex])
    more_single_dict = dict(zip(nosingle_list,test_pred))
    single_fitness_dict.update(more_single_dict)

    target_list = test_df['log_fitness']
    pred_list = []
    for mutants in test_df['mutant']:
        mutant1 = mutants.split(sep)[0]
        mutant2 = mutants.split(sep)[1]
        fitness = single_fitness_dict[mutant1]+single_fitness_dict[mutant2]
        pred_list.append(fitness)
    test_df['pred'] = pred_list
    pred_list = np.array(pred_list)
    target_list = np.array(target_list)
    spearman_v = spearmanr(pred_list,target_list)
    
    return test_df,spearman_v

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name',action='store', required=True)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    train_df = pd.read_csv('/home/chenlin/directed_evolution/gvp/input_data/{}/{}_single.csv'.format(dataset_name,dataset_name))
    test_df = pd.read_csv('/home/chenlin/directed_evolution/gvp/input_data/{}/{}_muti.csv'.format(dataset_name,dataset_name))
    test_df,spearman_v = runModel(train_df, test_df)
    print('for dataset {}, the spearman of the additive model is {}'.format(dataset_name,spearman_v))