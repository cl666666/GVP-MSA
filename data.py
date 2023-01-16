import json
from tkinter.messagebox import RETRY
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
from typing import Sequence, Tuple, List, Union
import pandas as pd
from extract_esm_msa1b_rep import *

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
   
def get_seqlen_from_fasta(dataset):
    fasta_file = 'input_data/{}/{}.fasta'.format(dataset,dataset)
    with open(fasta_file,'r') as f:
        lines = f.readlines()
    seq = lines[1].strip()
    ids = lines[0]
    offset = ids.split('/')[1].split('_')[0]
    return len(seq),seq,int(offset)

class ProteinGraphDataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, data_df, coords,seq_bind_pad,dataset_name,pad_msa = True,
                 num_positional_embeddings=16,get_msa_info = True,if_category=False,
                 top_k=15, num_rbf=16, device="cuda:0"):
        
        super(ProteinGraphDataset, self).__init__()
        self.get_msa_info = get_msa_info
        self.pad_msa = pad_msa
        self.if_category = if_category
        self.device_run_esmmsa = device
        self.top_k = top_k
        self.dataset_name = dataset_name
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.seqlen,self.wt_seq,self.offset = get_seqlen_from_fasta(dataset_name)
        self.data_df = self.get_mut_seq(data_df,self.wt_seq,'msa_seq')
        self.data_df = self.get_mut_seq(data_df,seq_bind_pad,'coords_seq')

        CHARS = ["-", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
         "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

        self.letter_to_num = {c: i for i, c in enumerate(CHARS)}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

        self.coords =coords
        self.additional_node = len(seq_bind_pad)-len(self.wt_seq)
        self.coords_info = self._get_coords_feature()

        self.wt_graph = self._get_wt_graph()

    def get_mut_seq(self,data_df,wt_seq,column_name):
        seq_list = []
        for i in range(len(data_df)):
            line = data_df.iloc[i]
            mutants = line['mutant']
            if mutants == 'WT':
                seq_list.append(wt_seq)
            else:
                seq_mut = wt_seq
                for mutant in mutants.split('-'):
                    mut_idx = int(mutant[1:-1])-self.offset
                    assert wt_seq[mut_idx] == mutant[0], ValueError('wild type seq is not consistent with mutant type')
                    seq_mut = seq_mut[:mut_idx] + mutant[-1] + seq_mut[mut_idx+1:]
                seq_list.append(seq_mut)
        data_df[column_name] = seq_list
        return data_df

    def __len__(self): 
        return len(self.data_df)
    def _get_coords_feature(self):
        coords = self.coords[:,:3,:] # seqlen,4,3
        with torch.no_grad():
            coords = torch.as_tensor(coords, 
                                     device=self.device, dtype=torch.float32)   

            mask = torch.isfinite(coords.sum(dim=(1,2))) # seqlen
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1,:] # seqlen,3
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            
            dihedrals = self._dihedrals(coords)                     
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))

        return X_ca,node_s, node_v, edge_s, edge_v,edge_index,mask
    def _get_esm_msa_rep(self):
        a2m_file_path = 'input_data/{}/{}.a2m'.format(self.dataset_name,self.dataset_name)
        msa_rep = get_esm_msa1b_rep(a2m_path=a2m_file_path,num_seqs=512,device=self.device_run_esmmsa)
        return msa_rep.to('cpu').squeeze(0) # 263, 768
    def _get_wt_graph(self):
        line = self.data_df.iloc[0]
        assert line['mutant'] == 'WT'
        coords_seq_list = [self.letter_to_num[a] for a in line['coords_seq']]
        coords_seq_tensor = torch.as_tensor(coords_seq_list,
                                  device=self.device, dtype=torch.long)
        msa_seq_tensor = torch.as_tensor([self.letter_to_num[a] for a in line['msa_seq']],
                                  device=self.device, dtype=torch.long)
        X_ca,node_s, node_v, edge_s, edge_v,edge_index,mask = self.coords_info

        wt_graph = torch_geometric.data.Data(target = line['log_fitness'],
                                         mutant = line['mutant'],
                                         seq = coords_seq_tensor,
                                         msa_seq = msa_seq_tensor,
                                         node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v,
                                         edge_index=edge_index, mask=mask,
                                         dataset_name = line['dataset_name'],
                                        )
        if self.get_msa_info:
            msa_rep = self._get_esm_msa_rep()
            if self.additional_node !=0:
                msa_rep = F.pad(msa_rep,(0,0,0,self.additional_node))
            wt_graph.msa_rep = msa_rep,
        if self.if_category:
            wt_graph.target_category =  line['category_2class']

        return wt_graph

    def __getitem__(self, i): 
        line = self.data_df.iloc[i]
        graph = self._featurize_as_graph(line)
        return graph,self.wt_graph
    def _featurize_as_graph(self,line):
        is_mut_site = torch.zeros(len(line['coords_seq']))
        mutants = line['mutant']
        if mutants !='WT':
            for mutant in mutants.split('-'):
                mut_idx = int(mutant[1:-1])-self.offset
                is_mut_site[mut_idx] = 1
        
        coords_seq_list = [self.letter_to_num[a] for a in line['coords_seq']]

        seq_tensor = torch.as_tensor(coords_seq_list,
                                  device=self.device, dtype=torch.long)
        
        data = torch_geometric.data.Data(target = line['log_fitness'],
                                         seq = seq_tensor,
                                         mutant = line['mutant'],
                                         dataset_name = line['dataset_name'],
                                        )
        if self.if_category:
            data.target_category =  line['category_2class']

        return data
                                
    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])  #seqlen*3,3
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = abs(edge_index[0] - edge_index[1])
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])#X.shape = seqlen,3
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)
    def _betac_orient(self, ca, cb):
        betac_orient = _normalize(ca - cb)#X.shape = seqlen,3
        return betac_orient

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 

