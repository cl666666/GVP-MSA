import sys
import torch
import os 
import pandas as pd
import math
from sklearn.utils import shuffle
import numpy as np
import math
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, ndcg_score

import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_solvent
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
from typing import Sequence, Tuple, List

def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        print('pred value is almost same,var is {}'.format(np.var(y_pred)))
        return 0.0
    return spearmanr(y_pred, y_true).correlation


def ndcg_old(y_pred, y_true):
    y_true_normalized = (y_true - y_true.mean()) / (y_true.std()+0.0000001)
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))
def ndcg(y_pred, y_true):
    min_ytrue = np.min(y_true)
    if min_ytrue <0:
        y_true = y_true + abs(min_ytrue)
    k = math.floor(len(y_true)*0.01)
    return ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1),k=k)

def aucroc(y_pred, y_true, y_cutoff):
    y_true_bin = (y_true >= y_cutoff)
    return roc_auc_score(y_true_bin, y_pred, average='micro')


class Logger(object):
    """Writes both to file and terminal"""
    def __init__(self, savepath, mode='a'):
        self.terminal = sys.stdout
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.log = open(os.path.join(savepath, 'logfile.log'), mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def randomSeed(random_seed):
    """Given a random seed, this will help reproduce results across runs"""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

class EarlyStopping(object):
    def __init__(self, 
            patience=100, eval_freq=1, best_score=None, 
            delta=1e-9, higher_better=True):
        self.patience = patience
        self.eval_freq = eval_freq
        self.best_score = best_score
        self.delta = delta
        self.higher_better = higher_better
        self.counter = 0
        self.early_stop = False
    
    def not_improved(self, val_score):
        if np.isnan(val_score):
            return True
        if self.higher_better:
            return val_score < self.best_score + self.delta
        else:
            return val_score > self.best_score - self.delta
    
    def update(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            is_best = True
        elif self.not_improved(val_score):
            self.counter += self.eval_freq
            if (self.patience is not None) and (self.counter > self.patience):
                self.early_stop = True
            is_best = False
        else:
            self.best_score = val_score
            self.counter = 0
            is_best = True
        return is_best

    # create train/val/test dataset separately
  # split_method 0: split randomly , 1: site-specific,
def get_splited_data(dataset_name,data_split_method,suffix = '',
         train_ratio=0.7,val_ratio=0.1,test_ratio=0.2,folder_num = 5,data_dir_prefix = ''):
    if data_split_method == 0:
        splitdatas = []
        assert train_ratio+val_ratio+test_ratio == 1
        datafile = './input_data/{}/{}{}.csv'.format(dataset_name,dataset_name,suffix)
        alldata = pd.read_csv(os.path.join(data_dir_prefix,datafile))
        sample_len = len(alldata)
        for fold_idx in range(folder_num):
            alldata_shffled = shuffle(alldata,random_state=fold_idx)
            val_size    = math.floor(val_ratio * sample_len)
            test_size   = math.floor(test_ratio * sample_len)

            test_dfs   = alldata_shffled[:test_size]
            wt_df = test_dfs.query("mutant=='WT'")
            test_dfs.drop(wt_df.index,inplace=True)
            test_dfs.reset_index(drop=True,inplace=True)

            val_dfs    = alldata_shffled[test_size:test_size + val_size]
            train_dfs  = alldata_shffled[test_size+val_size:]

            train_final = pd.concat([alldata.iloc[:1],train_dfs])
            test_dfs = pd.concat([alldata.iloc[:1],test_dfs])
            val_dfs = pd.concat([alldata.iloc[:1],val_dfs])

            train_final['dataset_name'] = dataset_name
            test_dfs['dataset_name'] = dataset_name
            val_dfs['dataset_name'] = dataset_name
            splitdatas.append((train_final,val_dfs,test_dfs))
        return splitdatas
    elif data_split_method == 1:
        splitdatas = []
        for fold_idx in range(folder_num):

            datadir = './input_data/{}/based_resid_split_data{}/fold_{}'.format(dataset_name,suffix,fold_idx)
            train = pd.read_csv(os.path.join(data_dir_prefix,datadir,'train.csv'))
            test = pd.read_csv(os.path.join(data_dir_prefix,datadir,'test.csv'))
            val = pd.read_csv(os.path.join(data_dir_prefix,datadir,'val.csv'))
            train['dataset_name'] = dataset_name
            test['dataset_name'] = dataset_name
            val['dataset_name'] = dataset_name
            splitdatas.append((train,val,test))
        return splitdatas

    else:
        raise ValueError('split data method is valid')

def get_fold_data(i,splitdatas_list):
    # get fold i test/train/val
    test_df_list = [splitdatas[i][2] for splitdatas in splitdatas_list]
    train_df_list = [splitdatas[i][0] for splitdatas in splitdatas_list]
    val_df_list = [splitdatas[i][1] for splitdatas in splitdatas_list]
    return (pd.concat(train_df_list),pd.concat(val_df_list),pd.concat(test_df_list))


def get_whole_structure(fpath):
    """
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    issolvent = filter_solvent(structure)
    structure = structure[~issolvent]
    chains = get_chains(structure)
    print(f'Found {len(chains)} chains:', chains, '\n')
    if len(chains) == 0:
        raise ValueError('No chains found in the input file.')

    structure = structure[structure.hetero == False]
    return structure,chains
def load_structure(fpath, chain=None,bind_chains=None):
    """
    Returns:
        biotite.structure.AtomArray
    """
    structure,chains = get_whole_structure(fpath)
    assert chain in chains, ValueError('target chain {} not found in pdb file'.format(chain))
    structure_target = structure[structure.chain_id == chain]
    structure_binds = []
    if bind_chains is not None and bind_chains is not False:
        for bind_chain in bind_chains:
            assert bind_chain in chains, ValueError('bind chain {} not found in pdb file'.format(bind_chain))
            structure_bind = structure[structure.chain_id == bind_chain]
            structure_binds.append(structure_bind)
    
    return structure_target, structure_binds

def extract_coords_from_structure(structure):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C", "CB"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities if r in ProteinSequence._dict_3to1.keys()])

    return coords, seq


def load_coords(fpath, chain,bind_chains=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    structure1,structure_binds = load_structure(fpath, chain,bind_chains=bind_chains)
    coords,seq = extract_coords_from_structure(structure1)
    coords_binds = []
    seq_binds = []
    for structure_bind in structure_binds:
        coords_bind,seq_bind = extract_coords_from_structure(structure_bind)
        coords_binds.append(coords_bind)
        seq_binds.append(seq_bind)
    return coords,seq,coords_binds,seq_binds

def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


def get_coords_seq(pdbfile,config,ifbindchain=True,ifbetac=False):
    chain = config['target_chain']
    bind_chains = config['bindding_chain']
    addition_chain = []
    if ifbindchain and bind_chains:
        addition_chain.extend(bind_chains)
    coords, wt_seq,coords_binds,seq_binds = load_coords(pdbfile, chain,bind_chains=addition_chain)
    assert coords.shape[0] == len(wt_seq)
    seqs = []
    seqs.append(wt_seq)
    seqs.extend(seq_binds)
    seq_pad = '-'*10
    seq_bind_pad = seq_pad.join(seqs)
    coord_out = coords
    for i in coords_binds:
        coord_out = coord_cat(coord_out,i)
    if not ifbetac:
        coord_out = coord_out[:,:3,:]

    return coord_out,seq_bind_pad

def coord_cat(coord1,coord2):
    coord_pad = np.zeros((10, 4, 3))
    coord_pad[:] = np.inf
    coords_binds_pad = []
    coords_binds_pad.append(coord1)
    coords_binds_pad.append(coord_pad)
    coords_binds_pad.append(coord2)
    coords_binds_pad = np.vstack(coords_binds_pad)
    return coords_binds_pad
