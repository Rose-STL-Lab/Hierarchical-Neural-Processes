import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:1")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

class L2_MLP_Encoder(nn.Module):

    def __init__(self, 
            in_dim, 
            out_dim,
            hidden_dim,
            zw_dim):

        nn.Module.__init__(self)

        layers = [nn.Conv2d(in_dim, hidden_dim, 3, stride=1, padding=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1), nn.ELU(), nn.BatchNorm2d(hidden_dim)]

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim*zw_dim*zw_dim, out_dim*zw_dim*zw_dim)
        self.cov_out = nn.Linear(hidden_dim*zw_dim*zw_dim, out_dim*zw_dim*zw_dim)
        self.cov_m = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x)
        mean = self.mean_out(output.view(x.shape[0],-1))
        cov = 0.1+ 0.9*self.cov_m(self.cov_out(output.view(x.shape[0],-1)))

        return mean.view(x.shape[0],-1,6,6), cov.view(x.shape[0],-1,6,6)

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


class Model(nn.Module):
    def __init__(self, logger, **model_kwargs):
        super().__init__()

        self.z_dim = int(model_kwargs.get('z_dim',32))
        self.zw_dim = int(model_kwargs.get('zw_dim',14))
        self.input_dim = int(model_kwargs.get('input_dim', 3)) #fully connected, 50+3
        self.output_dim = int(model_kwargs.get('output_dim', 50))
        self.hidden_dim = int(model_kwargs.get('hidden_dim', 32))
        self.encoder_output_dim = self.z_dim
        self.decoder_input_dim = self.z_dim + self.hidden_dim
        self.context_percentage_low = float(model_kwargs.get('context_percentage_low', 0.2))
        self.context_percentage_high = float(model_kwargs.get('context_percentage_high', 0.5))
        self.l2_encoder_model = L2_MLP_Encoder(self.hidden_dim*2, self.encoder_output_dim, self.hidden_dim, self.zw_dim)
        self.l2_decoder_model = L2_MLP_Decoder(self.z_dim + self.hidden_dim*2, self.output_dim, self.hidden_dim)

        self.l1_encoder_embed = L1_MLP_Embed(self.input_dim,self.hidden_dim, self.hidden_dim, self.zw_dim)
        self.l2_encoder_embed = L2_MLP_Embed(self.input_dim+self.output_dim, self.hidden_dim, self.hidden_dim, self.zw_dim)
        self.l1_decoder_embed = L1_MLP_Embed(self.input_dim,self.hidden_dim, self.hidden_dim, self.zw_dim)
        self.l2_decoder_embed = L2_MLP_Embed(self.input_dim,self.hidden_dim, self.hidden_dim, self.zw_dim)

        self._logger = logger


    def split_context_target(self, y_low, x, y, context_percentage_low, context_percentage_high, level):
        """Helper function to split randomly into context and target"""
        context_percentage = np.random.uniform(context_percentage_low,context_percentage_high)

        n_context = int(x.shape[0]*context_percentage)
        ind = np.arange(x.shape[0])
        mask = np.random.choice(ind, size=n_context, replace=False)
        others = np.delete(ind,mask)


        return y_low[mask], x[mask], y[mask], y_low[others], x[others], y[others]


    def sample_z(self, mean, var, n=1):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(var.data.new(n,var.size(0),var.size(1),var.size(2)).normal_()).to(device)

        std = torch.sqrt(var)
        return torch.unsqueeze(mean, dim=0) + torch.unsqueeze(std, dim=0) * eps

    def xy_to_r(self, y_low, x, y, level):

        if level == 2:
            y1_low = self.l1_encoder_embed(y_low)
            r1 = self.l2_encoder_embed(torch.cat([x, y],dim=-3))
            r_mu, r_cov = self.l2_encoder_model(torch.cat([y1_low, r1],dim=-3))

        return r_mu, r_cov

    def z_to_y(self, y_low, x, zs, level):

        outputs = []

        if level == 2:
            y1_low = self.l1_decoder_embed(y_low)
            x1 = self.l2_decoder_embed(x)
            output = self.l2_decoder_model(torch.cat([y1_low,x1,zs], dim=-3))

        return output

    def ba_z_agg(self, r_mu, r_cov, z_mu, z_cov):

        # r_mu = torch.swapaxes(r_mu,0,1)
        # r_cov = torch.swapaxes(r_cov,0,1)

        v = r_mu - z_mu
        w_cov_inv = 1 / r_cov
        z_cov_new = 1 / (1 / z_cov + torch.sum(w_cov_inv, dim=0))
        z_mu_new = z_mu + z_cov_new * torch.sum(w_cov_inv * v, dim=0)
        return z_mu_new, z_cov_new


    def forward(self, l1_y_all=None, l2_x_all=None, l2_y_all=None, batches_seen=None, test=False, l2_z_mu_all=None, l2_z_cov_all=None, l2_z_mu_c=None, l2_z_cov_c=None):

        if test==False:

            self._logger.debug("starting point complete, starting split source and target")
            #first half for context, second for target
            l1_y_c, l2_x_c,l2_y_c, l1_y_t, l2_x_t,l2_y_t = self.split_context_target(l1_y_all, l2_x_all,l2_y_all, self.context_percentage_low, self.context_percentage_high, level=2)
            self._logger.debug("data split complete, starting encoder")
            #l2_encoder
            l2_r_mu_all, l2_r_cov_all = self.xy_to_r(l1_y_all, l2_x_all, l2_y_all, level=2)
            l2_r_mu_c, l2_r_cov_c = self.xy_to_r(l1_y_c, l2_x_c, l2_y_c, level=2)
            l2_z_mu_all, l2_z_cov_all = self.ba_z_agg(l2_r_mu_all,l2_r_cov_all,l2_z_mu_all,l2_z_cov_all)
            l2_z_mu_c, l2_z_cov_c = self.ba_z_agg(l2_r_mu_c,l2_r_cov_c,l2_z_mu_all,l2_z_cov_all)


            #sample z
            l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_t.size(0))
            #l2_decoder
            self._logger.debug("Encoder complete, starting decoder")
            l2_output_mu, l2_output_cov = self.z_to_y(l1_y_t, l2_x_t,l2_zs, level=2)

            l2_truth = l2_y_t

            self._logger.debug("Decoder complete")
            if batches_seen == 0:
                self._logger.info(
                    "Total trainable parameters {}".format(count_parameters(self))
                )

            return l2_output_mu, l2_output_cov, l2_truth, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c
            
        else:
            l2_output_mu, l2_output_cov = None, None
            if l2_x_all is not None:
                l2_zs = self.sample_z(l2_z_mu_all, l2_z_cov_all, l2_x_all.size(0))
                l2_output_mu, l2_output_cov = self.z_to_y(l1_y_all, l2_x_all, l2_zs, level=2)


            return l2_output_mu, l2_output_cov

