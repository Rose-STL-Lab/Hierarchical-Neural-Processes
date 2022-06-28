import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.model import Model
from model.pytorch.loss import nll_loss
from model.pytorch.loss import nll_metric
from model.pytorch.loss import mae_metric
from model.pytorch.loss import nonormalized_mae_metric
from model.pytorch.loss import kld_gaussian_loss
from torch.utils.tensorboard import SummaryWriter

import csv


device = torch.device("cuda:2")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Supervisor:
    def __init__(self, random_seed, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # logging.
        self._log_dir = self._get_log_dir(kwargs, self.random_seed)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        # self.l1_y_scaler = self._data['l1_y_scaler']
        # self.l2_y_scaler = self._data['l2_y_scaler']

        self.input_dim = int(self._model_kwargs.get('input_dim', 3))
        self.output_dim = int(self._model_kwargs.get('output_dim', 50))
        self.z_dim = int(self._model_kwargs.get('z_dim',32))
        self.l1_num_nodes = int(self._model_kwargs.get('l1_num_nodes', 18))
        self.l2_num_nodes = int(self._model_kwargs.get('l2_num_nodes', 85))
        self.batch_size = int(self._data_kwargs.get('batch_size', 128))
        self.seq_len = int(self._model_kwargs.get('seq_len', 50))  # for the encoder
        self.num_batches = None #int(0)

        # setup model
        model = Model(self._logger, **self._model_kwargs)
        self.model = model.cuda(device) if torch.cuda.is_available() else model

        self.l1_z_mu_all = torch.zeros(self.l1_num_nodes, self.z_dim).to(device)
        self.l1_z_cov_all = torch.ones(self.l1_num_nodes, self.z_dim).to(device)
        self.l2_z_mu_all = torch.zeros(self.l2_num_nodes, self.z_dim).to(device)
        self.l2_z_cov_all = torch.ones(self.l2_num_nodes, self.z_dim).to(device)
        self.l1_z_mu_c = torch.zeros(self.l1_num_nodes, self.z_dim).to(device)
        self.l1_z_cov_c = torch.ones(self.l1_num_nodes, self.z_dim).to(device)
        self.l2_z_mu_c = torch.zeros(self.l2_num_nodes, self.z_dim).to(device)
        self.l2_z_cov_c = torch.ones(self.l2_num_nodes, self.z_dim).to(device)
        
        
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs, random_seed):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            # max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            # num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            # rnn_units = kwargs['model'].get('rnn_units')
            # structure = '-'.join(
                # ['%d' % rnn_units for _ in range(num_rnn_layers)])
            # horizon = kwargs['model'].get('horizon')

            run_id = 'model_lr_%g_bs_%d_%s_%d/' % (
                learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'), random_seed)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch, l1_z_total, l2_z_total, outputs, saved_model, val_nll, nll, mae, non_mae):
        if not os.path.exists('weights'):
            os.makedirs('weights')
        if not os.path.exists('meanstd_mfhnp_results'):
            os.makedirs('meanstd_mfhnp_results')

        config = dict(self._kwargs)
        config['model_state_dict'] = saved_model
        config['epoch'] = epoch
        torch.save(config, 'weights/model_seed%d_epo%d.tar' % (self.random_seed, epoch))
        torch.save(l1_z_total, 'weights/l1_z_seed%d_epo%d.tar' % (self.random_seed, epoch))
        torch.save(l2_z_total, 'weights/l2_z_seed%d_epo%d.tar' % (self.random_seed, epoch))
        np.savez_compressed('meanstd_mfhnp_results/test_seed%d.npz'% (self.random_seed), **outputs)
        self._logger.info("Saved model at {}".format(epoch))

        with open(r'meanstd_mfhnp_results/metric_seed%d.csv'% (self.random_seed), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([nll, non_mae])

        return 'weights/model_seed%d_epo%d.tar' % (self.random_seed, epoch)


    def load_model(self):
        assert os.path.exists('weights/l1_z_seed%d_epo%d.tar' % (self.random_seed, self._epoch_num)), 'Z for low fidelity at epoch %d not found' % self._epoch_num
        assert os.path.exists('weights/l2_z_seed%d_epo%d.tar' % (self.random_seed, self._epoch_num)), 'Z for high fidelity at epoch %d not found' % self._epoch_num
        checkpoint1 = torch.load('weights/l1_z_seed%d_epo%d.tar' % (self.random_seed, self._epoch_num), map_location='cpu')
        checkpoint2 = torch.load('weights/l2_z_seed%d_epo%d.tar' % (self.random_seed, self._epoch_num), map_location='cpu')
        self.l1_z_mu_all = checkpoint1[0].to(device)
        self.l1_z_cov_all = checkpoint1[1].to(device)
        self.l2_z_mu_all = checkpoint2[0].to(device)
        self.l2_z_cov_all = checkpoint2[1].to(device)
        assert os.path.exists('weights/model_seed%d_epo%d.tar' % (self.random_seed, self._epoch_num)), 'Weights for high fidelity at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('weights/model_seed%d_epo%d.tar' % (self.random_seed, self._epoch_num), map_location='cpu')


        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)


    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.model = self.model.eval()

            #change val_iterator
            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()

            y_truths = []
            y_preds_mu = []
            y_preds_cov = []

            for _, (x, y, adj_mx) in enumerate(val_iterator):
                x, y, adj_mx = self._test_prepare_data(x, y, adj_mx)
                _, _, output_mu, output_cov = self.model(l2_x_all=x, l2_adj_all=adj_mx, test=True, l2_z_mu_all= self.l2_z_mu_all, l2_z_cov_all=self.l2_z_cov_all)
                y_truths.append(y.cpu())
                y_preds_mu.append(output_mu.cpu())
                y_preds_cov.append(output_cov.cpu())

            y_preds_mu = np.concatenate(y_preds_mu, axis=0)
            y_preds_cov = np.concatenate(y_preds_cov, axis=0)
            y_truths = np.concatenate(y_truths, axis=0) 

            # y_preds_mu = y_preds_mu[:34*30] #15*30
            # y_preds_cov = y_preds_cov[:34*30]
            # y_truths = y_truths[:34*30]

            # l2_std = self.l2_y_scaler.std
            # y_truths_scaled = self.l2_y_scaler.inverse_transform(y_truths)
            # y_preds_mu_scaled = self.l2_y_scaler.inverse_transform(y_preds_mu)
            # y_preds_cov_scaled = y_preds_cov * l2_std * l2_std

            nll_metric, mae_metric, nonormalized_mae_metric = self._test_loss(y_preds_mu, y_preds_cov, y_truths)
            self._writer.add_scalar('{} loss'.format(dataset), nll_metric, batches_seen) #check

            return nll_metric, mae_metric, nonormalized_mae_metric, {'pred_mu': y_preds_mu, 'pred_cov': y_preds_cov, 'truth': y_truths}

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,gamma=lr_decay_ratio)

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        self.num_batches = self._data['l2_train_loader'].num_batch
        self._logger.info("num_batches:{}".format(self.num_batches))

        batches_seen = self.num_batches * self._epoch_num

        # with autograd.detect_anomaly():
        saved_model = dict()
        saved_output = dict()
        saved_nll = None
        saved_mae = None
        saved_non_mae = None
        saved_epoch = None
        saved_l1_z_total = None
        saved_l2_z_total = None
        saved_val_nll = None

        # check data size
        l1_data_size = self._data['l1_train_loader'].size
        l2_data_size = self._data['l2_train_loader'].size
        print("l1_data_size:",l1_data_size, "l2_data_size:",l2_data_size)

        for epoch_num in range(self._epoch_num, epochs):
            # reshuffle the data
            self._data = utils.load_dataset(**self._data_kwargs)

            self.model = self.model.train()

            l1_train_iterator = self._data['l1_train_loader'].get_iterator()
            l2_train_iterator = self._data['l2_train_loader'].get_iterator()

            losses = []
            l1_nll_losses = []
            l2_nll_losses = []
            l1_kld_losses = []
            l2_kld_losses = []

            self.l1_z_mu_all = torch.zeros(self.l1_num_nodes, self.z_dim).to(device)
            self.l1_z_cov_all = torch.ones(self.l1_num_nodes, self.z_dim).to(device)
            self.l2_z_mu_all = torch.zeros(self.l2_num_nodes, self.z_dim).to(device)
            self.l2_z_cov_all = torch.ones(self.l2_num_nodes, self.z_dim).to(device)
            self.l1_z_mu_c = torch.zeros(self.l1_num_nodes, self.z_dim).to(device)
            self.l1_z_cov_c = torch.ones(self.l1_num_nodes, self.z_dim).to(device)
            self.l2_z_mu_c = torch.zeros(self.l2_num_nodes, self.z_dim).to(device)
            self.l2_z_cov_c = torch.ones(self.l2_num_nodes, self.z_dim).to(device)


            start_time = time.time()


            for index, ((l1_x, l1_y, l1_adj), (l2_x, l2_y, l2_adj)) in enumerate(zip(l1_train_iterator, l2_train_iterator)): # need to be fixed
                optimizer.zero_grad()

                l1_x, l1_y, l1_adj = self._train_l1_prepare_data(l1_x, l1_y, l1_adj)
                l2_x, l2_y, l2_adj = self._train_l2_prepare_data(l2_x, l2_y, l2_adj)


                l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l1_truth, l2_truth, self.l1_z_mu_all, self.l1_z_cov_all, self.l1_z_mu_c, self.l1_z_cov_c, self.l2_z_mu_all, self.l2_z_cov_all, self.l2_z_mu_c, self.l2_z_cov_c = self.model(l1_x, l1_y, l1_adj, l2_x, l2_y, l2_adj, batches_seen, False, self.l1_z_mu_all, self.l1_z_cov_all, self.l1_z_mu_c, self.l1_z_cov_c, self.l2_z_mu_all, self.l2_z_cov_all, self.l2_z_mu_c, self.l2_z_cov_c)

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr, eps=epsilon)
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,gamma=lr_decay_ratio)

                l1_nll_loss, l2_nll_loss, l1_kld_loss, l2_kld_loss = self._compute_loss(l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l1_truth, l2_truth, self.l1_z_mu_all, self.l1_z_cov_all, self.l1_z_mu_c, self.l1_z_cov_c, self.l2_z_mu_all, self.l2_z_cov_all, self.l2_z_mu_c, self.l2_z_cov_c)
                loss = l1_nll_loss + l1_kld_loss + l2_nll_loss + l2_kld_loss

                self._logger.debug(loss.item())

                losses.append(loss.item())
                l1_nll_losses.append(l1_nll_loss.item())
                l2_nll_losses.append(l2_nll_loss.item())
                l1_kld_losses.append(l1_kld_loss.item())
                l2_kld_losses.append(l2_kld_loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

                self.l1_z_mu_all = self.l1_z_mu_all.detach()
                self.l1_z_cov_all = self.l1_z_cov_all.detach()
                self.l2_z_mu_all = self.l2_z_mu_all.detach()
                self.l2_z_cov_all = self.l2_z_cov_all.detach()
                self.l1_z_mu_c = self.l1_z_mu_c.detach()
                self.l1_z_cov_c = self.l1_z_cov_c.detach()
                self.l2_z_mu_c = self.l2_z_mu_c.detach()
                self.l2_z_cov_c = self.l2_z_cov_c.detach()


            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

            _, _, val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)
            end_time = time.time()

            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, l1_nll: {:.4f}, l1_kld: {:.4f}, l2_nll: {:.4f}, l2_kld: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), np.mean(l1_nll_losses), np.mean(l1_kld_losses), np.mean(l2_nll_losses), np.mean(l2_kld_losses), val_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_nll_loss, test_mae_loss, test_non_mae_loss, test_outputs = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) test_nll: {:.4f}, test_mae: {:.4f}, test_non_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           test_nll_loss, test_mae_loss, test_non_mae_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)


            if val_loss < min_val_loss:
                wait = 0
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}.'
                    .format(min_val_loss, val_loss))
                min_val_loss = val_loss

                saved_model = self.model.state_dict()
                saved_val_nll = val_loss
                saved_nll = test_nll_loss
                saved_mae = test_mae_loss
                saved_non_mae = test_non_mae_loss
                saved_outputs = test_outputs
                saved_epoch = epoch_num
                saved_l1_z_total = torch.stack([self.l1_z_mu_all, self.l1_z_cov_all], dim=0)
                saved_l2_z_total = torch.stack([self.l2_z_mu_all, self.l2_z_cov_all], dim=0)

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    model_file_name = self.save_model(saved_epoch, saved_l1_z_total, saved_l2_z_total, saved_outputs, saved_model, saved_val_nll, saved_nll, saved_mae, saved_non_mae)

                    self._logger.info(
                        'Final Val loss {:.4f}, Test NLL loss {:.4f}, Test MAE loss {:.4f}, Test Non MAE loss {:.4f}, '
                        'saving to {}'.format(saved_val_nll, saved_nll, saved_mae, saved_non_mae, model_file_name))
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

            if epoch_num == epochs-1:
                model_file_name = self.save_model(saved_epoch, saved_l1_z_total, saved_l2_z_total, saved_outputs, saved_model, saved_val_nll, saved_nll, saved_mae, saved_non_mae)

                self._logger.info(
                    'Final Val loss {:.4f}, Test NLL loss {:.4f}, Test MAE loss {:.4f}, Test Non MAE loss {:.4f}, '
                    'saving to {}'.format(saved_val_nll, saved_nll, saved_mae,saved_non_mae, model_file_name))

    # def _test_prepare_data(self, x, y, adj_mx):

    #     x = torch.from_numpy(x).float()
    #     y = torch.from_numpy(y).float()
    #     adj_mx = torch.from_numpy(adj_mx).float()
    #     self._logger.debug("X: {}".format(x.size()))
    #     self._logger.debug("y: {}".format(y.size()))
    #     self._logger.debug("adj_mx: {}".format(adj_mx.size()))
    #     x = x.permute(1, 0, 2, 3)
    #     y = y.permute(1, 0, 2, 3)
    #     adj_mx = adj_mx.permute(1, 0, 2, 3)
    #     return x.to(device), y.to(device), adj_mx.to(device)

    def _test_prepare_data(self, x, y, adj_mx):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        adj_mx = torch.from_numpy(adj_mx).float()
        x = x.reshape(-1,self.l2_num_nodes,self.input_dim)
        y = y.reshape(-1,self.l2_num_nodes,self.output_dim)
        adj_mx = adj_mx.reshape(-1,self.l2_num_nodes,self.l2_num_nodes)
        return x.to(device), y.to(device), adj_mx.to(device)

    def _train_l1_prepare_data(self, x, y, adj_mx):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        adj_mx = torch.from_numpy(adj_mx).float()
        x = x.reshape(-1,self.l1_num_nodes,self.input_dim)
        y = y.reshape(-1,self.l1_num_nodes,self.output_dim)
        adj_mx = adj_mx.reshape(-1,self.l1_num_nodes,self.l1_num_nodes)
        return x.to(device), y.to(device), adj_mx.to(device)

    def _train_l2_prepare_data(self, x, y, adj_mx):

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        adj_mx = torch.from_numpy(adj_mx).float()
        x = x.reshape(-1,self.l2_num_nodes,self.input_dim)
        y = y.reshape(-1,self.l2_num_nodes,self.output_dim)
        adj_mx = adj_mx.reshape(-1,self.l2_num_nodes,self.l2_num_nodes)
        return x.to(device), y.to(device), adj_mx.to(device)

    def _compute_loss(self, l1_output_mu, l1_output_cov, l2_output_mu, l2_output_cov, l1_truth, l2_truth, l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c):
        
        # l1_mean = torch.tensor(self.l1_y_scaler.mean).to(device)
        # l1_std = torch.tensor(self.l1_y_scaler.std).to(device)
        # l2_mean = torch.tensor(self.l2_y_scaler.mean).to(device)
        # l2_std = torch.tensor(self.l2_y_scaler.std).to(device)

        # l1_output_mu = (l1_output_mu*l1_std) + l1_mean
        # l2_output_mu = (l2_output_mu*l2_std) + l2_mean
        # l1_output_cov = l1_output_cov*l1_std*l1_std
        # l2_output_cov = l2_output_cov*l2_std*l2_std
        # l1_truth = (l1_truth*l1_std) + l1_mean
        # l2_truth = (l2_truth*l2_std) + l2_mean

        return nll_loss(l1_output_mu, l1_output_cov, l1_truth), nll_loss(l2_output_mu, l2_output_cov, l2_truth), kld_gaussian_loss(l1_z_mu_all, l1_z_cov_all, l1_z_mu_c, l1_z_cov_c), kld_gaussian_loss(l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c)

    def _test_loss(self, y_predicted_mu, y_predicted_cov, y_truth):

        return nll_metric(y_predicted_mu, y_predicted_cov, y_truth), mae_metric(y_predicted_mu, y_truth), nonormalized_mae_metric(y_predicted_mu, y_truth)

