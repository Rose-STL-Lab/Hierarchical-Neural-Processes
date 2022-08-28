import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGCNConv

device = torch.device("cuda:2")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MLP_Encoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        x = torch.cat([x,adj],dim=-1)
        output = self.model(x)
        mean = self.mean_out(output)
        return mean

class L1_MLP_ZEncoder(nn.Module):
    """Takes an r representation and produces the mean & variance of the 
    normally distributed function encoding, z."""
    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Sigmoid()


    def forward(self, inputs):
        output = self.model(inputs)
        mean = self.mean_out(output)
        cov = 0.1+0.9*self.cov_m(self.cov_out(output))

        return mean, cov

class L2_MLP_ZEncoder(nn.Module):
    """Takes an r representation and produces the mean & variance of the 
    normally distributed function encoding, z."""
    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Sigmoid()


    def forward(self, inputs, z):
        inputs = torch.cat([inputs, z], dim=-1)
        output = self.model(inputs)
        mean = self.mean_out(output)
        cov = 0.1+0.9*self.cov_m(self.cov_out(output))

        return mean, cov

class MLP_Decoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim, 
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        # layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Softplus()

    def forward(self, x, adj):
        x = torch.cat([x,adj],dim=-1)
        output = self.model(x)
        mean = self.mean_out(output)
        cov = self.cov_m(self.cov_out(output))
        # cov = torch.exp(self.cov_out(output))

        return mean, cov

class MLP_Z1Z2_Encoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_layers=2,
            hidden_dim=32):

        nn.Module.__init__(self)
        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            # layers.append(nn.LayerNorm(hidden_dim))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Sigmoid()
        # self.cov_m = nn.ELU()

    def forward(self, x, y):
        x = torch.swapaxes(x, -2, -1)
        y = torch.swapaxes(y, -2, -1)
        output = self.model(torch.cat([x, y],dim=-1))
        mean = self.mean_out(output)
        cov = 0.1+ 0.9*self.cov_m(self.cov_out(output))
        mean = torch.swapaxes(mean, -2, -1)
        cov = torch.swapaxes(cov, -2, -1)

        return mean, cov


class Model(nn.Module):
    def __init__(self, logger, **model_kwargs):
        super().__init__()

        self.hidden_layers = int(model_kwargs.get('hidden_layers',5))
        self.z_hidden_layers = int(model_kwargs.get('z_hidden_layers',1))
        self.z_hidden_dim = int(model_kwargs.get('z_hidden_dim', 32))
        self.z_dim = int(model_kwargs.get('z_dim',32))
        self.input_dim = int(model_kwargs.get('input_dim', 3))
        self.output_dim = int(model_kwargs.get('output_dim', 50))
        self.hidden_dim = int(model_kwargs.get('hidden_dim', 32))
        self.encoder_output_dim = self.z_dim
        self.decoder_input_dim = self.z_dim + self.input_dim
        self.l1_num_nodes = int(model_kwargs.get('l1_num_nodes', 18))
        self.l2_num_nodes = int(model_kwargs.get('l2_num_nodes', 85))
        self.context_percentage_low = float(model_kwargs.get('context_percentage_low', 0.2))
        self.context_percentage_high = float(model_kwargs.get('context_percentage_high', 0.5))
        self.l1_encoder_model = MLP_Encoder(self.input_dim+self.output_dim+self.l1_num_nodes, self.encoder_output_dim, self.hidden_layers, self.hidden_dim)
        self.l2_encoder_model = MLP_Encoder(self.input_dim+self.output_dim+self.l2_num_nodes, self.encoder_output_dim, self.hidden_layers, self.hidden_dim)
        self.l1_decoder_model = MLP_Decoder(self.decoder_input_dim+self.l1_num_nodes, self.output_dim, self.hidden_layers, self.hidden_dim)
        self.l2_decoder_model = MLP_Decoder(self.decoder_input_dim+self.l2_num_nodes, self.output_dim, self.hidden_layers, self.hidden_dim)

        self.l1_z_encoder_model = L1_MLP_ZEncoder(self.z_dim, self.z_dim, self.z_hidden_layers, self.z_hidden_dim)
        self.l2_z_encoder_model = L2_MLP_ZEncoder(self.z_dim+self.z_dim, self.z_dim, self.z_hidden_layers, self.z_hidden_dim)
        self.z2_z1_agg = MLP_Z1Z2_Encoder(self.l1_num_nodes*2, self.l2_num_nodes)

        self._logger = logger


    def split_context_target(self, x, y, adj, context_percentage_low, context_percentage_high, level):
        """Helper function to split randomly into context and target"""
        context_percentage = np.random.uniform(context_percentage_low,context_percentage_high)
        if level == 1:
            node_dim = 18
        elif level == 2:
            node_dim = 85
        x = x.reshape(-1,node_dim,x.shape[-1])
        y = y.reshape(-1,node_dim,y.shape[-1])
        adj= adj.reshape(-1,node_dim,node_dim)

        n_context = int(x.shape[0]*context_percentage)
        ind = np.arange(x.shape[0])
        mask = np.random.choice(ind, size=n_context, replace=False)
        others = np.delete(ind,mask)


        return x[mask], y[mask], adj[mask], x[others], y[others], adj[others]


    def sample_z(self, mean, var, n=1):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(var.data.new(n,var.size(0),var.size(1)).normal_()).to(device)

        std = torch.sqrt(var)
        return torch.unsqueeze(mean, dim=0) + torch.unsqueeze(std, dim=0) * eps

    def xy_to_r(self, x, y, adj, level):
        if level == 1:
            r_mu = self.l1_encoder_model(torch.cat([x, y],dim=-1), adj)
        elif level == 2:
            r_mu = self.l2_encoder_model(torch.cat([x, y],dim=-1), adj)

        return r_mu

    def z_to_y(self, x, zs, adj, level):

        outputs = []

        if level == 1:
            output = self.l1_decoder_model(torch.cat([x,zs], dim=-1), adj)

        elif level == 2:
            output = self.l2_decoder_model(torch.cat([x,zs], dim=-1), adj)

        return output

    def mean_z_agg(self, r, level, z_mu=None):

        if level == 1:
            r_agg = torch.mean(r,dim=0)
            z_mu, z_cov = self.l1_z_encoder_model(r_agg)
        if level == 2:
            r_agg = torch.mean(r,dim=0)
            z_mu, z_cov = self.l2_z_encoder_model(r_agg, z_mu)

        return z_mu, z_cov


    def forward(self, l1_x_all=None, l1_y_all=None, l1_adj_all=None, l2_x_all=None, l2_y_all=None, l2_adj_all=None, batches_seen=None, test=False, l1_z_mu_all=None, l1_z_cov_all=None, l1_z_mu_c=None, l1_z_cov_c=None, l2_z_mu_all=None, l2_z_cov_all=None, l2_z_mu_c=None, l2_z_cov_c=None):
        """
        seq2seq forward pass
        :param node: shape (seq_len, batch_size, num_age_bracket input_dim)
        :param inputs0: shape (batch_size, input_dim)
        :param labels: shape (horizon, batch_size, output)
        :param batches_seen: batches seen till now
        :param test: train or test
        :param z_mean_all: z_mean_all for the last training epoch
        :param z_var_all: z_var_all for the last training epoch
        :return: outputs,truth: (self.horizon, batch_size, self.output_dim)
                 z_mean_all, z_var_all, z_mean_context, z_var_context (self.horizon, batch_size, z_dim)
        """

        """
        train:
        split context and target points for both graph_low, graph_high
        l2_encoder_model: (x2_c,e2_c,y2_c)-> r2-> z2
        l1_encoder_model: (x1_c,e1_c,y1_c)-> r1 
        bayesian aggregation between two fidelity levels: (z2, r1) -> z1
        l2_decoder_model: (x2_t,e2_t,z2)-> y2_t
        l1_decoder_model: (x1_t,e1_t,z1)-> y1_t
        test:
        rollout, saved z1 and z2
        l2_decoder_model: (x2_t,e2_t,z2)-> y1_2
        l1_decoder_model: (x1_t,e1_t,z1)-> y1_t
        """
        if test==False:

            self._logger.debug("starting point complete, starting split source and target")
            #first half for context, second for target
            l1_x_c,l1_y_c,l1_adj_c,l1_x_t,l1_y_t,l1_adj_t = self.split_context_target(l1_x_all,l1_y_all,l1_adj_all, self.context_percentage_low, self.context_percentage_high, level=1)
            l2_x_c,l2_y_c,l2_adj_c,l2_x_t,l2_y_t,l2_adj_t = self.split_context_target(l2_x_all,l2_y_all,l2_adj_all, self.context_percentage_low, self.context_percentage_high, level=2)
            self._logger.debug("data split complete, starting encoder")
            #l1_encoder
            l1_r_all = self.xy_to_r(l1_x_all, l1_y_all, l1_adj_all, level=1)
            l1_r_c = self.xy_to_r(l1_x_c, l1_y_c, l1_adj_c, level=1)
            l1_z_mu_all, l1_z_cov_all = self.mean_z_agg(l1_r_all,level=1)
            l1_z_mu_c, l1_z_cov_c = self.mean_z_agg(l1_r_c,level=1)

            # NN between two fidelity levels
            l2_z_mu_all_0,l2_z_cov_all_0 = self.z2_z1_agg(l1_z_mu_all, l1_z_cov_all)
            #l2_encoder
            l2_r_all = self.xy_to_r(l2_x_all, l2_y_all, l2_adj_all, level=2)
            l2_r_c = self.xy_to_r(l2_x_c, l2_y_c, l2_adj_c, level=2)
            l2_z_mu_all, l2_z_cov_all = self.mean_z_agg(l2_r_all,level=2,z_mu=l2_z_mu_all_0)
            l2_z_mu_c, l2_z_cov_c = self.mean_z_agg(l2_r_c,level=2,z_mu=l2_z_mu_all_0)

            #sample z
            l1_zs = self.sample_z(l1_z_mu_all, l1_z_cov_all, l1_x_t.size(0))
            l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_t.size(0))
            #l1_decoder, l2_decoder
            self._logger.debug("Encoder complete, starting decoder")
            l1_output_mu, l1_output_cov = self.z_to_y(l1_x_t,l1_zs, l1_adj_t, level=1)
            l2_output_mu, l2_output_cov = self.z_to_y(l2_x_t,l2_zs, l2_adj_t, level=2)

            l1_truth = l1_y_t
            l2_truth = l2_y_t

            self._logger.debug("Decoder complete")
            if batches_seen == 0:
                self._logger.info(
                    "Total trainable parameters {}".format(count_parameters(self))
                )

            return l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l1_truth, l2_truth, l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c
            
        else:
            l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov = None, None, None, None
            if l1_x_all is not None:
                l1_zs = self.sample_z(l1_z_mu_all, l1_z_cov_all, l1_x_all.size(0))
                l1_output_mu, l1_output_cov = self.z_to_y(l1_x_all, l1_zs, l1_adj_all, level=1)
            if l2_x_all is not None:
                l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_all.size(0))
                l2_output_mu, l2_output_cov = self.z_to_y(l2_x_all, l2_zs, l2_adj_all, level=2)


            return l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov


