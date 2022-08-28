import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:2")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class L1_MLP_Encoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_dim,
            zw_dim):

        nn.Module.__init__(self)

        layers = [nn.Conv2d(in_dim, int(hidden_dim/2), 3, stride=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.Conv2d(int(hidden_dim/2), int(hidden_dim/2), 3, stride=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.Conv2d(int(hidden_dim/2), hidden_dim, 3, stride=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim*zw_dim*zw_dim, out_dim*zw_dim*zw_dim)

    def forward(self, x):
        output = self.model(x)
        mean = self.mean_out(output.view(x.shape[0],-1))

        return mean.view(x.shape[0],-1,6,6)

class L1_MLP_Embed(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_dim,
            zw_dim):

        nn.Module.__init__(self)

        layers = [nn.Conv2d(in_dim, int(hidden_dim/2), 3, stride=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.Conv2d(int(hidden_dim/2), int(hidden_dim/2), 3, stride=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.Conv2d(int(hidden_dim/2), hidden_dim, 3, stride=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]

        self.model = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim*zw_dim*zw_dim, out_dim*zw_dim*zw_dim)

    def forward(self, x):
        output = self.model(x)
        output = self.out(output.view(x.shape[0],-1))

        return output.view(x.shape[0],-1,6,6)

class L1_MLP_Decoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim, 
            hidden_dim):

        nn.Module.__init__(self)

        layers = [nn.ConvTranspose2d(in_dim, hidden_dim, 3, stride=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, stride=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.ConvTranspose2d(hidden_dim, int(hidden_dim/2), 3, stride=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.ConvTranspose2d(int(hidden_dim/2), int(hidden_dim/2), 3, stride=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]

        
        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(int(hidden_dim/2), out_dim)
        self.cov_out = nn.Linear(int(hidden_dim/2), out_dim)
        self.cov_m = nn.Softplus()

    def forward(self, x):
        output = self.model(x)
        output = torch.movedim(output,-3,-1)
        mean = self.mean_out(output)
        cov = self.cov_m(self.cov_out(output))

        return torch.movedim(mean,-1,-3), torch.movedim(cov,-1,-3)

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
        inputs = torch.movedim(inputs,-3,-1)
        output = self.model(inputs)
        mean = self.mean_out(output)
        cov = 0.1+0.9*self.cov_m(self.cov_out(output))

        return torch.movedim(mean,-1,-3), torch.movedim(cov,-1,-3)

class L2_MLP_Encoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_dim,
            zw_dim):

        nn.Module.__init__(self)

        layers = [nn.Conv2d(in_dim, int(hidden_dim/2), 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.Conv2d(int(hidden_dim/2), int(hidden_dim/2), 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.Conv2d(int(hidden_dim/2), hidden_dim, 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim*zw_dim*zw_dim, out_dim*zw_dim*zw_dim)

    def forward(self, x):
        output = self.model(x)
        mean = self.mean_out(output.view(x.shape[0],-1))

        return mean.view(x.shape[0],-1,6,6)

class L2_MLP_Embed(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_dim,
            zw_dim):

        nn.Module.__init__(self)

        layers = [nn.Conv2d(in_dim, int(hidden_dim/2), 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.Conv2d(int(hidden_dim/2), int(hidden_dim/2), 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.Conv2d(int(hidden_dim/2), hidden_dim, 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]

        self.model = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim*zw_dim*zw_dim, out_dim*zw_dim*zw_dim)

    def forward(self, x):
        output = self.model(x)
        output = self.out(output.view(x.shape[0],-1))

        return output.view(x.shape[0],-1,6,6)

class L2_MLP_Decoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim, 
            hidden_dim):

        nn.Module.__init__(self)

        layers = [nn.ConvTranspose2d(in_dim, hidden_dim, 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, stride=2, padding=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.ConvTranspose2d(hidden_dim, int(hidden_dim/2), 3, stride=2, padding=0), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]
        layers += [nn.ConvTranspose2d(int(hidden_dim/2), int(hidden_dim/2), 3, stride=2, padding=0), nn.ELU(), nn.BatchNorm2d(int(hidden_dim/2))]

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(int(hidden_dim/2), out_dim)
        self.cov_out = nn.Linear(int(hidden_dim/2), out_dim)
        self.cov_m = nn.Softplus()

    def forward(self, x):
        output = self.model(x)
        output = torch.movedim(output,-3,-1)
        mean = self.mean_out(output)
        cov = self.cov_m(self.cov_out(output))

        return torch.movedim(mean,-1,-3), torch.movedim(cov,-1,-3)

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
        inputs = torch.cat([inputs, z], dim=-3)
        inputs = torch.movedim(inputs,-3,-1)
        output = self.model(inputs)
        mean = self.mean_out(output)
        cov = 0.1+0.9*self.cov_m(self.cov_out(output))

        return torch.movedim(mean,-1,-3), torch.movedim(cov,-1,-3)

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

    def forward(self, x):
        x = torch.movedim(x,-3,-1)
        output = self.model(x)
        mean = self.mean_out(output)
        cov = 0.1+ 0.9*self.cov_m(self.cov_out(output))

        return torch.movedim(mean,-1,-3), torch.movedim(cov,-1,-3)


class Model(nn.Module):
    def __init__(self, logger, **model_kwargs):
        super().__init__()

        self.z_hidden_layers = int(model_kwargs.get('z_hidden_layers',1))
        self.z_hidden_dim = int(model_kwargs.get('z_hidden_dim', 32))
        self.z_dim = int(model_kwargs.get('z_dim',32))
        self.zw_dim = int(model_kwargs.get('zw_dim',14))
        self.input_dim = int(model_kwargs.get('input_dim', 3))
        self.output_dim = int(model_kwargs.get('output_dim', 50))
        self.hidden_dim = int(model_kwargs.get('hidden_dim', 32))
        self.encoder_output_dim = self.z_dim
        self.decoder_input_dim = self.z_dim + self.hidden_dim
        self.context_percentage_low = float(model_kwargs.get('context_percentage_low', 0.2))
        self.context_percentage_high = float(model_kwargs.get('context_percentage_high', 0.5))
        self.l1_encoder_model = L1_MLP_Encoder(self.input_dim+self.output_dim, self.encoder_output_dim, self.hidden_dim, self.zw_dim)
        self.l2_encoder_model = L2_MLP_Encoder(self.input_dim+self.output_dim, self.encoder_output_dim, self.hidden_dim, self.zw_dim)
        self.l1_decoder_model = L1_MLP_Decoder(self.decoder_input_dim, self.output_dim, self.hidden_dim)
        self.l2_decoder_model = L2_MLP_Decoder(self.decoder_input_dim, self.output_dim, self.hidden_dim)
        self.l1_decoder_embed = L1_MLP_Embed(self.input_dim,self.hidden_dim, self.hidden_dim, self.zw_dim)
        self.l2_decoder_embed = L2_MLP_Embed(self.input_dim,self.hidden_dim, self.hidden_dim, self.zw_dim)

        self.l1_z_encoder_model = L1_MLP_ZEncoder(self.z_dim, self.z_dim, self.z_hidden_layers, self.z_hidden_dim)
        self.l2_z_encoder_model = L2_MLP_ZEncoder(self.z_dim+self.z_dim, self.z_dim, self.z_hidden_layers, self.z_hidden_dim)
        self.z2_z1_agg = MLP_Z1Z2_Encoder(self.z_dim, self.z_dim)

        self._logger = logger


    def split_context_target(self, x, y, context_percentage_low, context_percentage_high, level):
        """Helper function to split randomly into context and target"""
        context_percentage = np.random.uniform(context_percentage_low,context_percentage_high)

        n_context = int(x.shape[0]*context_percentage)
        ind = np.arange(x.shape[0])
        mask = np.random.choice(ind, size=n_context, replace=False)
        others = np.delete(ind,mask)


        return x[mask], y[mask], x[others], y[others]


    def sample_z(self, mean, var, n=1):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(var.data.new(n,var.size(0),var.size(1),var.size(2)).normal_()).to(device)

        std = torch.sqrt(var)
        return torch.unsqueeze(mean, dim=0) + torch.unsqueeze(std, dim=0) * eps

    def xy_to_r(self, x, y, level):
        if level == 1:
            r_mu = self.l1_encoder_model(torch.cat([x, y],dim=-3))
        elif level == 2:
            r_mu = self.l2_encoder_model(torch.cat([x, y],dim=-3))

        return r_mu

    def z_to_y(self, x, zs, level):

        outputs = []

        if level == 1:
            x1 = self.l1_decoder_embed(x)
            output = self.l1_decoder_model(torch.cat([x1,zs], dim=-3))

        elif level == 2:
            x1 = self.l2_decoder_embed(x)
            output = self.l2_decoder_model(torch.cat([x1,zs], dim=-3))

        return output

    def mean_z_agg(self, r, level, z_mu=None):

        if level == 1:
            r_agg = torch.mean(r,dim=0)
            z_mu, z_cov = self.l1_z_encoder_model(r_agg)
        if level == 2:
            r_agg = torch.mean(r,dim=0)
            z_mu, z_cov = self.l2_z_encoder_model(r_agg, z_mu)

        return z_mu, z_cov


    def forward(self, l1_x_all=None, l1_y_all=None, l2_x_all=None, l2_y_all=None, batches_seen=None, test=False, l1_z_mu_all=None, l1_z_cov_all=None, l1_z_mu_c=None, l1_z_cov_c=None, l2_z_mu_all=None, l2_z_cov_all=None, l2_z_mu_c=None, l2_z_cov_c=None):

        if test==False:

            self._logger.debug("starting point complete, starting split source and target")
            #first half for context, second for target
            l1_x_c,l1_y_c,l1_x_t,l1_y_t = self.split_context_target(l1_x_all,l1_y_all, self.context_percentage_low, self.context_percentage_high, level=1)
            l2_x_c,l2_y_c,l2_x_t,l2_y_t = self.split_context_target(l2_x_all,l2_y_all, self.context_percentage_low, self.context_percentage_high, level=2)
            self._logger.debug("data split complete, starting encoder")
            #l1_encoder
            l1_r_all = self.xy_to_r(l1_x_all, l1_y_all, level=1)
            l1_r_c = self.xy_to_r(l1_x_c, l1_y_c, level=1)
            l1_z_mu_all, l1_z_cov_all = self.mean_z_agg(l1_r_all,level=1)
            l1_z_mu_c, l1_z_cov_c = self.mean_z_agg(l1_r_c,level=1)

            # NN between two fidelity levels
            l2_z_mu_all_0,l2_z_cov_all_0 = self.z2_z1_agg(l1_z_mu_all)
            #l2_encoder
            l2_r_all = self.xy_to_r(l2_x_all, l2_y_all, level=2)
            l2_r_c = self.xy_to_r(l2_x_c, l2_y_c, level=2)
            l2_z_mu_all, l2_z_cov_all = self.mean_z_agg(l2_r_all,level=2,z_mu=l2_z_mu_all_0)
            l2_z_mu_c, l2_z_cov_c = self.mean_z_agg(l2_r_c,level=2,z_mu=l2_z_mu_all_0)

            #sample z
            l1_zs = self.sample_z(l1_z_mu_all, l1_z_cov_all, l1_x_t.size(0))
            l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_t.size(0))
            #l1_decoder, l2_decoder
            self._logger.debug("Encoder complete, starting decoder")
            l1_output_mu, l1_output_cov = self.z_to_y(l1_x_t,l1_zs, level=1)
            l2_output_mu, l2_output_cov = self.z_to_y(l2_x_t,l2_zs, level=2)

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
                l1_output_mu, l1_output_cov = self.z_to_y(l1_x_all, l1_zs, level=1)
            if l2_x_all is not None:
                l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_all.size(0))
                l2_output_mu, l2_output_cov = self.z_to_y(l2_x_all, l2_zs, level=2)


            return l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov

