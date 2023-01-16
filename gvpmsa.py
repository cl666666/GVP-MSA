import sys, time,os, random,copy
import collections
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleDict
from torch_geometric.loader import DataLoader

from utils import *
from data import *
from model_utils import GVP, GVPConvLayer, LayerNorm


class GVPMSA(object):
    def __init__(self,
            output_dir,
            dataset_names,
            train_dfs_dict,
            val_dfs_dict,
            test_dfs_dict,
            dataset_config,
            pdb_path_prefix = '',
            device='cuda:0',
            node_in_dim = (6, 3),
            node_h_dim = (100,16),
            edge_in_dim = (32, 1),
            edge_h_dim =(32,1),
            lr = 1e-4,
            batch_size=128,
            top_k=15,data_category=False,
             multi_train=False,out_dim=1,
            n_ensembles=3,load_model_path = None,
            esm_msa_linear_hidden=128, num_layers=2,msa_in=True,
             drop_rate=0.1):
        if data_category:
            assert out_dim == 3
        else:
            assert out_dim ==1
        self.batch_size = batch_size
        self.dataset_names = dataset_names
        self.top_k = top_k
        self.data_category = data_category
        self.output_dir = output_dir
        self.device = device
        self.msa_in = msa_in

        self.coords_dict = self.get_coords_dict(pdb_path_prefix,dataset_config)
        self.data_loader_dict = self.get_dataloader(train_dfs_dict = train_dfs_dict,
                                              val_dfs_dict = val_dfs_dict,
                                              test_dfs_dict = test_dfs_dict)

        self.logger = Logger(output_dir)
        if load_model_path:
            model_dict = torch.load(load_model_path,map_location=self.device)
            model = model_dict['model']
            model.load_state_dict(model_dict['model_para'])
            model.multi_train = False
            self.models = [model]

        else:
            self.models = [
            VEPModel(node_in_dim=node_in_dim, node_h_dim=node_h_dim, 
                     edge_in_dim=edge_in_dim, edge_h_dim=edge_h_dim,dataset_names=dataset_names,
                     multi_train = multi_train,
                    esm_msa_linear_hidden = esm_msa_linear_hidden,
                 seq_in=True, num_layers=num_layers, drop_rate=drop_rate,
                 out_dim = out_dim,seq_esm_msa_in=msa_in)
            .to(self.device) for _ in range(n_ensembles)]

        weight = torch.tensor([1,100],dtype=torch.float,device=device)
        self.Loss_c = nn.CrossEntropyLoss(weight = weight)
        self.Loss_mse = nn.MSELoss()
        self.batch_size = batch_size
        self.optimizers = [torch.optim.Adam(model.parameters(),lr=lr) for model in self.models]
        self._test_pack = None
    def get_coords_dict(self,pdb_path_prefix,dataset_config):
        coords_dict = {}
        for dataset_ in self.dataset_names:
            pdbfile = os.path.join(pdb_path_prefix,'{}/{}.pdb'.format(dataset_,dataset_))
            coords_binds_pad,seq_bind_pad = get_coords_seq(pdbfile,dataset_config[dataset_],ifbindchain=True,ifbetac=False)
            coords_dict[dataset_] = (coords_binds_pad,seq_bind_pad)
        return coords_dict
    def get_dataloader(self, train_dfs_dict,val_dfs_dict,test_dfs_dict):
        train_loader_dict = {}
        val_loader_dict = {}
        test_loader_dict = {}
        for dataset_name in train_dfs_dict.keys():
            train_dataset = ProteinGraphDataset(train_dfs_dict[dataset_name],self.coords_dict[dataset_name][0],
                 self.coords_dict[dataset_name][1],
                 dataset_name,get_msa_info = self.msa_in,top_k=self.top_k,if_category=self.data_category,device=self.device)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,)
            train_loader_dict[dataset_name] = train_loader

        for dataset_name in val_dfs_dict.keys():
            val_dataset = ProteinGraphDataset(val_dfs_dict[dataset_name],self.coords_dict[dataset_name][0],
                 self.coords_dict[dataset_name][1],
                 dataset_name,get_msa_info = self.msa_in,top_k=self.top_k,if_category=self.data_category,device=self.device)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            val_loader_dict[dataset_name] = val_loader

        for dataset_name in test_dfs_dict.keys():
            test_dataset = ProteinGraphDataset(test_dfs_dict[dataset_name],self.coords_dict[dataset_name][0],
                 self.coords_dict[dataset_name][1],
                 dataset_name,get_msa_info = self.msa_in,top_k=self.top_k,if_category=self.data_category,device=self.device)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader_dict[dataset_name] = test_loader
        return {'train':train_loader_dict,'val':val_loader_dict,'test':test_loader_dict}

    def train_onefold(self, fold_idx,epochs=1000,patience=150, save_checkpoint=False, save_prediction=True):
        pred_list_ensemble = 0
        for midx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            stopper = EarlyStopping(patience=patience,higher_better=True)

            for epoch in range(epochs):
                # Training
                losses_all,(pred_list,target_list),spearman_v_train = self.runModel(model,optimizer,mode='train')
                self.logger.write('Epoch{},train total loss: {},  Spearman:{}\n'.format(epoch,
                     losses_all,spearman_v_train))
    
                # Validation
                losses_all,(pred_list,target_list),spearman_v_val = self.runModel(model,optimizer,mode='val')
                self.logger.write('Epoch{},val total loss: {},  Spearman:{}\n'.format(epoch,
                     losses_all,spearman_v_val))
    
                if epoch% 20 == 19:
                    self.logger.flush()
                
                spearman_v_val_mean = sum(spearman_v_val)/len(spearman_v_val)
                is_best = stopper.update(spearman_v_val_mean)
                
                if is_best:
                    best_model_para = model.state_dict()
                    best_epoch = epoch
                    # test
                    losses_all,(pred_list,target_list),spearman_v_test = self.runModel(model,optimizer,mode='test')
                    self.logger.write('Epoch{},test total loss: {},  Spearman is {},\n'.format(epoch,
                         losses_all,spearman_v_test))
                    best_pred_target_test = (pred_list,target_list)
                    best_val_spearman = spearman_v_val_mean
                    best_test_spearman = spearman_v_test

                if stopper.early_stop or epoch == epochs-1:
                    self.logger.write('Early stop at epoch {}\n'.format(epoch))
                    self.logger.write(
                        'ensemble idx {}, best epoch {}, best validation spearman is {},best test spearman is {},\n\n'.format(
                            midx,best_epoch, best_val_spearman,best_test_spearman))
                    break
    
            pred_list_ensemble += np.array(best_pred_target_test[0])
            if save_checkpoint:
                best_stat = {'data_fold':fold_idx,'model_para':best_model_para,
                'model':model,
                'epoch':best_epoch,'test_pred_target':best_pred_target_test,
                'best_test_metrics':best_test_spearman,
                'best_val_metrics':best_val_spearman,
                }
                torch.save(best_stat, os.path.join(self.output_dir,'model_fold{}_ensemble{}.pt'.format(fold_idx,midx))
                )
        if save_prediction:
            dataframe = pd.DataFrame({'pred':pred_list_ensemble,'target':best_pred_target_test[1]})
            dataframe.to_csv(os.path.join(self.output_dir,'pred_fold{}.csv'.format(fold_idx)))
        ensemble_metrics_spearman = spearman(pred_list_ensemble,best_pred_target_test[1])
        ensemble_metrics_ndcg = ndcg(pred_list_ensemble,best_pred_target_test[1])
        self.logger.write('ensemble {} models, fold {}, spearman is {}, ndcg is {}\n'.format(
                   len(self.models),fold_idx,ensemble_metrics_spearman,ensemble_metrics_ndcg))
        return dataframe

    def runModel(self, model, optimizer,mode='test'):
        device = self.device

        losses_all = 0
        if mode == 'train':
            data_loader_dict = self.data_loader_dict[mode]
            datasets_list = data_loader_dict.keys()

            spearman_list = []
            count = 0
            target_list = []
            pred_list = []
            data_loader_dict_iter = {}
            for dataset in datasets_list:
                data_loader_dict_iter[dataset] = iter(data_loader_dict[dataset])
            while True:
                dataset = random.sample(datasets_list,1)[0]
                try:
                    (graph,wt_graph) = next(data_loader_dict_iter[dataset])
                except StopIteration:
                    break
    
                count +=1
        
                model.train()
                out = model(graph.to(device),wt_graph.to(device))
                target = graph.target.float().to(device)
    
                if self.data_category:
                    out_classfy,out_reg = out
                    target_category = torch.tensor(graph.target_category,dtype=torch.long,device=device)
                    loss_classfy = self.Loss_c(out_classfy,target_category)
    
                else:
                    out_reg = out
                    loss_classfy = 0
                loss_reg = self.Loss_mse(out_reg,target)
                loss = loss_classfy + loss_reg
                loss.backward()
    
                losses_all += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                target_list.extend(target.cpu().detach().numpy())
                pred_list.extend(out_reg.cpu().detach().numpy())
            pred_list = np.vstack(pred_list)[:,0]
            target_list = np.vstack(target_list)[:,0]
            spearman_v = spearman(pred_list,target_list)
        
            return losses_all/count,(pred_list,target_list),spearman_v
        elif mode == 'test' or 'val':
            
            data_loader_dict = self.data_loader_dict[mode]
            datasets_list = data_loader_dict.keys()

            count = 0
            losses_all = 0
            spearman_list = []
            outall_reg_all = []
            target_all_all = []
            for dataset in datasets_list:
                target_list = []
                pred_list = []
    
                for (graph,wt_graph) in data_loader_dict[dataset]:
                    with torch.no_grad():
                        model.eval()
                        count +=1
                        out = model(graph.to(device),wt_graph.to(device))
                        target = graph.target.float().to(device)
                        if self.data_category:
                            out_classfy,out_reg = out
                            target_category = torch.tensor(graph.target_category,dtype=torch.long,device=device)
                            loss_classfy = self.Loss_c(out_classfy,target_category)
            
                        else:
                            out_reg = out
                            loss_classfy = 0
                        target_list.extend(target.cpu().detach().numpy())
                        pred_list.extend(out_reg.cpu().detach().numpy())
                        loss_reg = self.Loss_mse(out_reg,target)
                        loss = loss_classfy + loss_reg
    
                        losses_all += loss.item()
                pred_list = np.vstack(pred_list)
                target_list = np.vstack(target_list)
                spearman_list.append(spearman(pred_list,target_list))
    
                outall_reg_all.append(pred_list)
                target_all_all.append(target_list)
            
            return_out = (losses_all/count),(np.vstack(outall_reg_all)[:,0],np.vstack(target_all_all)[:,0]),spearman_list
        return return_out

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class VEPModel(nn.Module):
 
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,dataset_names,
                 multi_train = False,
                 esm_msa_linear_hidden = 128,
                 seq_in=True, num_layers=2, drop_rate=0.1,seq_esm_msa_in=True,
                 out_dim = 3):
        
        super(VEPModel, self).__init__()
        self.node_h_dim = node_h_dim
        self.seq_esm_msa_in = seq_esm_msa_in
        self.out_dim = out_dim
        self.esm_msa_linear = nn.Linear(768,esm_msa_linear_hidden)
        self.multi_train = multi_train

        if seq_esm_msa_in:
            node_in_dim = (node_in_dim[0] + esm_msa_linear_hidden, node_in_dim[1])

        if seq_in:
            self.W_s = nn.Embedding(21, 20)
            node_in_dim = (node_in_dim[0] + 20*3, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))
            
        self.dense = nn.Sequential(
            nn.Linear(ns, ns//2), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(ns//2, ns*2)
        )

        self.readout = nn.Sequential(
            AggregateLayer(d_model = ns*2),
            GlobalPredictor(d_model = ns*2,
                                d_h=128, d_out=out_dim)
        )
        if multi_train:
            readout_list = [copy.deepcopy(self.readout) for i in range(len(dataset_names))]
            self.readout_dict = ModuleDict(dict(zip(dataset_names,readout_list)))

    def forward(self,graph,wt_graph):
        out = self.forward1(graph,wt_graph)
        out = self.dense(out)
        if self.multi_train:
            out = self.readout_dict[graph.dataset_name[0]](out)
        else:
            out = self.readout(out)
            
        if self.out_dim ==3:
            return out[:,:2],out[:,2]
        elif self.out_dim ==1:
            return out[:,0]
        elif self.out_dim ==4:
            return out[:,:3],out[:,3]
        else:
            print('out dim not in [0,3], not implement')

    def forward1(self, graph,graph_wt):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        batch_num = graph_wt.batch[-1]+1
        h_V = (graph_wt.node_s,graph_wt.node_v)
        h_E = (graph_wt.edge_s,graph_wt.edge_v)
        seq = graph.seq
        seq_wt = graph_wt.seq
        edge_index = graph_wt.edge_index
        if seq is not None:
            seq = self.W_s(seq.long())
            seq_wt = self.W_s(seq_wt.long())
            h_V = (torch.cat([h_V[0], seq,seq_wt,seq-seq_wt], dim=-1), h_V[1])

        if self.seq_esm_msa_in: #[h_V[0].shape = (bs*seqlen,dim)
            h_V = (torch.cat([h_V[0], self.esm_msa_linear(graph_wt.msa_rep[0])], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layerid,layer in enumerate(self.layers):
            h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)
        hidden_dim = out.shape[-1]
        out = out.reshape(batch_num,-1,hidden_dim)
        
        return out

class AggregateLayer(nn.Module):
    def __init__(self, d_model=None, dropout=0.1):
        super(AggregateLayer, self).__init__()
        self.attn = nn.Sequential(collections.OrderedDict([
            ('layernorm', nn.LayerNorm(d_model)),
            ('fc', nn.Linear(d_model, 1, bias=False)),
            ('dropout', nn.Dropout(dropout)),
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, context):

        weight = self.attn(context) 
        output = torch.bmm(context.transpose(-1, -2), weight)
        output = output.squeeze(-1)
        return output



class GlobalPredictor(nn.Module):
    def __init__(self, d_model=None, d_h=None, d_out=None, dropout=0.5):
        super(GlobalPredictor, self).__init__()
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.predict_layer = nn.Sequential(collections.OrderedDict([
            # ('batchnorm', nn.BatchNorm1d(d_model)),
            ('fc1', nn.Linear(d_model, d_h)),
            ('tanh', nn.Tanh()),
            ('dropout', nn.Dropout(dropout)),
            ('fc2', nn.Linear(d_h, d_out))
        ]))

    def forward(self, x):
        if x.shape[0] !=1:
            x = self.batchnorm(x)
        x = self.predict_layer(x)
        return x
