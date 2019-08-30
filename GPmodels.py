#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:30:49 2018

@author: fsc
"""
import numpy as np
import pymc3 as pm
from scipy.interpolate import griddata, interp2d
import theano.tensor as tt
import sys
from matplotlib import pyplot as plt
from pyDOE import lhs
from sklearn.metrics import precision_recall_fscore_support
from theano.printing import Print
from theano.tensor.slinalg import solve_lower_triangular
from scipy import stats






def invlogit(x, eps=sys.float_info.epsilon):
    return (1.0 + 2.0 * eps) / (1.0 + np.exp(-x)) + eps

def normalize(X,lb,ub):
    return (X - lb)/(ub - lb)

def denormalize(X, lb, ub):
    return lb + X*(ub - lb)

def prettyplot(xlabel, ylabel, xlabelpad = -10, ylabelpad = -20, minXticks = True, minYticks = True):
    plt.xlabel(xlabel, labelpad = xlabelpad)
    plt.ylabel(ylabel, labelpad = ylabelpad)

    if minXticks:
        plt.xticks(plt.xlim())
        rang, labels = plt.xticks()
        labels[0].set_horizontalalignment("left")
        labels[-1].set_horizontalalignment("right")

    if minYticks:
        plt.yticks(plt.ylim())
        rang, labels = plt.yticks()
        labels[0].set_verticalalignment("bottom")
        labels[-1].set_verticalalignment("top")


class BaseClassifier(object):
    def __init__(self, lb, ub, sampling_func = None, X_test = None, Y_test = None):

        self.lb = lb
        self.ub = ub
        self.sampling_func = sampling_func

        self.h = 0.05

        self.X_next = None
        self.pred_samples_grid = None

        self.X_test = X_test

        self.last_id = 0

        if X_test is not None:
            self.X_test = normalize(X_test, lb, ub)
            self.error = []
            if Y_test is None:
                self.Y_true = self.sampling_func(X_test)
            else:
                self.Y_true = Y_test

    def create_model(self):
        raise "not implemented"

    def sample_model(self):
        with self.model:
            self.trace = pm.sample(1000, chains = 2, tune = 1000, nuts_kwargs = {'target_accept': 0.95})

    def sample_predictive(self,X_test, offset = 0, n_samples = 100):
        N = X_test.shape[0]
   #     Y_pred = np.array([])
        for i, X in enumerate(np.array_split(X_test, int(N / 500))):
            print(X.shape)
            self.last_id += 1
            varname = 'f_test_%i' % (self.last_id)
            pred_samples = self.generate_samples(varname, X, n_samples = n_samples)
            if i == 0:
                Y_pred = pred_samples[varname]
            else:
                Y_pred = np.concatenate((Y_pred, pred_samples[varname]), axis = 1)
    #        Y_pred = np.append(Y_pred,pred_samples[varname])
            print(pred_samples[varname].shape, Y_pred.shape)
        return Y_pred

    def generate_samples(self,name, X_new, n_samples = 500):
        raise "not implemented"

    def sample_grid(self):

        self.xx, self.yy = np.meshgrid(np.arange(0, 1 + self.h, self.h), np.arange(0, 1 + self.h, self.h))

        self.X_new = np.c_[self.xx.ravel(), self.yy.ravel()]

        self.pred_samples_grid = self.generate_samples('f_pred',  self.X_new)['f_pred']

    def plot(self, filename):
        raise "not implemented"

    def compute_next_point_cand(self):

        self.pred_samples_cand = self.sample_predictive(self.X_cand)

        ent = -np.abs(self.pred_samples_cand.mean(0))/(self.pred_samples_cand.std(0) + 1e-9)


        self.X_next = self.X_cand[ent.argmax()]
        print(self.X_next, ent.max())

    def append_next_point(self):
        raise "not implemented"

    def test_model(self):
        Y_test = self.sample_predictive(self.X_test)
        Y_test = np.round(invlogit(Y_test).mean(0))
        self.error.append([np.sum(np.abs(self.Y_true - Y_test)),precision_recall_fscore_support(self.Y_true, Y_test)])
        print('accuracy: ', 100.*self.error[-1][0]/self.X_test.shape[0])
        print(self.error[-1])

    def active_learning(self, N = 15, plot = False, filename = 'active_learning_%i.png'):
        for i in range(N):
            print('%i / %i' % (i+1,N))
            self.create_model()
            self.sample_model()
            if self.X_test is not None:
                self.test_model()
            self.compute_next_point_cand()
            self.append_next_point()
            if plot:
                self.plot(filename = filename % i, cand = True)



#%%

class BaseMFclassifier(BaseClassifier):
    def __init__(self,X_L, Y_L, X_H, Y_H, lb, ub,sampling_func, X_test = None, Y_test = None, N_cand = 1000, boundary_H = None, boundary_L = None):
        super(BaseMFclassifier, self).__init__(lb, ub, sampling_func = sampling_func, X_test = X_test, Y_test = Y_test)
        self.X_L = normalize(X_L, lb, ub)
        self.Y_L = Y_L
        self.X_H = normalize(X_H, lb, ub)
        self.Y_H = Y_H

        self.X = np.concatenate((X_L,X_H))
        self.Y = np.concatenate((Y_L,Y_H))

        self.dim = X_H.shape[1]

        self.X_cand = lhs(self.dim, N_cand)


    def plot(self,filename = 'MFGP.png', cand = False):
        assert self.dim == 2, 'can only plot 2D functions'
        if cand:
            prob_cand = invlogit(self.pred_samples_cand).mean(0)
            ent_cand = -np.abs(self.pred_samples_cand.mean(0))/(self.pred_samples_cand.std(0) + 1e-9)

            fig = plt.figure(1,figsize=(5.5,10))
            plt.clf()
            fig.set_size_inches(5.5,10)
            plt.subplot(211)
            cnt = plt.tricontourf(self.X_cand[:,0], self.X_cand[:,1], prob_cand,  np.linspace(0,1,100))
            for c in cnt.collections:
                c.set_edgecolor("face")

            cb = plt.colorbar(ticks = [0,1])
            cb.set_label('class probability [-]', labelpad = -10)

            labels = cb.ax.get_yticklabels()
            labels[0].set_verticalalignment("bottom")
            labels[-1].set_verticalalignment("top")

            plt.scatter(self.X_H[:-1,0],self.X_H[:-1,1],c=self.Y_H[:-1])

            if self.X_next is not None:
                plt.scatter([self.X_next[0]],[self.X_next[1]],color = 'k',marker = '*')

            plt.xlim([0,1])
            plt.ylim([0,1])

            if self.boundary_H is not None:
                hf = self.boundary_H(np.linspace(0,1,100)[:,None])
                plt.plot(np.linspace(0,1,100),hf,'k')
            if self.boundary_L is not None:
                lf = self.boundary_L(np.linspace(0,1,100)[:,None])
                plt.plot(np.linspace(0,1,100),lf,'k--')


            prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad = -10)

            plt.subplot(212)
            cnt = plt.tricontourf(self.X_cand[:,0], self.X_cand[:,1], ent_cand,100)

            for c in cnt.collections:
                c.set_edgecolor("face")
            plt.colorbar(label = 'active learning [-]', ticks = [])
            plt.scatter(self.X_H[:-1,0],self.X_H[:-1,1],c=self.Y_H[:-1])
            plt.xlim([0,1])
            plt.ylim([0,1])


            if self.X_next is not None:
                plt.scatter([self.X_next[0]],[self.X_next[1]],color = 'k',marker = '*')

            if self.boundary_H is not None:
                hf = self.boundary_H(np.linspace(0,1,100)[:,None])
                plt.plot(np.linspace(0,1,100),hf,'k')
            if self.boundary_L is not None:
                lf = self.boundary_L(np.linspace(0,1,100)[:,None])
                plt.plot(np.linspace(0,1,100),lf,'k--')


            prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad = -10)

        else:
            if self.pred_samples_grid is None:
                self.sample_grid()


            probs = invlogit(self.pred_samples_grid)
            prob = probs.mean(0)
            ent = np.abs(self.pred_samples_grid.mean(0))/(self.pred_samples_grid.std(0) + 1e-9)

            plt.figure(1,figsize=(5.5,10))
            plt.clf()
            plt.subplot(211)
            plt.contourf(self.xx, self.yy, np.reshape(prob, self.xx.shape))
            cb = plt.colorbar(ticks = [0,1])
            cb.set_label('class probability [-]', labelpad = -10)

            labels = cb.ax.get_yticklabels()
            labels[0].set_verticalalignment("bottom")
            labels[-1].set_verticalalignment("top")

            plt.scatter(self.X_H[:,0],self.X_H[:,1],c=self.Y_H)

            if self.X_next is not None:
                plt.scatter([self.X_next[0]],[self.X_next[1]],color = 'k',marker = '*')

            prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad = -10)

            plt.subplot(212)
            plt.contourf(self.xx, self.yy, np.reshape(ent, self.xx.shape))
            plt.colorbar(label = 'active learning [-]', ticks = [])
            plt.scatter(self.X_H[:,0],self.X_H[:,1],c=self.Y_H)


            if self.X_next is not None:
                plt.scatter([self.X_next[0]],[self.X_next[1]],color = 'k',marker = '*')

            prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad = -10)


        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)

    def append_next_point(self):
        self.X_H = np.vstack((self.X_H, self.X_next))
        Y_next = 1.0*self.sampling_func(denormalize(self.X_next[None,:], self.lb, self.ub))
        self.Y_H = np.concatenate((self.Y_H, np.array(Y_next)))

        self.X = np.concatenate((self.X_L,self.X_H))
        self.Y = np.concatenate((self.Y_L,self.Y_H))

class BaseSFclassifier(BaseClassifier):
    def __init__(self,X, Y, lb, ub,sampling_func, X_test = None, Y_test = None, N_cand = 1000, boundary = None):
        super(BaseSFclassifier, self).__init__(lb, ub, sampling_func = sampling_func, X_test = X_test, Y_test = Y_test)

        self.X = normalize(X, lb, ub)
        self.Y = Y
        self.dim = X.shape[1]

        self.X_cand = lhs(self.dim, N_cand)


    def plot(self,filename = 'GP.png', cand = False):
        assert self.dim == 2, 'can only plot 2D functions'

        if cand:
            prob_cand = invlogit(self.pred_samples_cand).mean(0)
            ent_cand = np.abs(self.pred_samples_cand.mean(0))/(self.pred_samples_cand.std(0) + 1e-9)

            plt.figure(1,figsize=(5.5,10))
            plt.clf()
            plt.subplot(211)
            cnt = plt.tricontourf(self.X_cand[:,0], self.X_cand[:,1], prob_cand,  np.linspace(0,1,100))
            for c in cnt.collections:
                c.set_edgecolor("face")

            cb = plt.colorbar(ticks = [0,1])
            cb.set_label('class probability [-]', labelpad = -10)

            labels = cb.ax.get_yticklabels()
            labels[0].set_verticalalignment("bottom")
            labels[-1].set_verticalalignment("top")

            plt.scatter(self.X[:-1,0],self.X[:-1,1],c=self.Y[:-1])

            if self.X_next is not None:
                plt.scatter([self.X_next[0]],[self.X_next[1]],color = 'k',marker = '*')

            plt.xlim([0,1])
            plt.ylim([0,1])

            if self.boundary is not None:
                hf = self.boundary(np.linspace(0,1,100)[:,None])
                plt.plot(np.linspace(0,1,100),hf,'k')

            prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad = -10)

            plt.subplot(212)
            cnt = plt.tricontourf(self.X_cand[:,0], self.X_cand[:,1], -ent_cand,100)
            for c in cnt.collections:
                c.set_edgecolor("face")
            plt.colorbar(label = 'active learning [-]', ticks = [])
            plt.scatter(self.X[:,0],self.X[:,1],c=self.Y)

            plt.xlim([0,1])
            plt.ylim([0,1])


            if self.X_next is not None:
                plt.scatter([self.X_next[0]],[self.X_next[1]],color = 'k',marker = '*')


            if self.boundary is not None:
                hf = self.boundary(np.linspace(0,1,100)[:,None])
                plt.plot(np.linspace(0,1,100),hf,'k')

            prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad = -10)



        else:
            if self.pred_samples_grid is None:
                self.sample_grid()

            probs = invlogit(self.pred_samples_grid)
            prob = probs.mean(0)
            ent = np.abs(self.pred_samples_grid.mean(0))/(self.pred_samples_grid.std(0) + 1e-9)

            plt.figure(1,figsize=(5.5, 10))
            plt.clf()
            plt.subplot(211)
            plt.contourf(self.xx, self.yy, np.reshape(prob, self.xx.shape))
            cb = plt.colorbar(ticks = [0,1])
            cb.set_label('class probability [-]', labelpad = -10)

            labels = cb.ax.get_yticklabels()
            labels[0].set_verticalalignment("bottom")
            labels[-1].set_verticalalignment("top")

            plt.scatter(self.X[:,0],self.X[:,1],c=self.Y)

            if self.X_next is not None:
                plt.scatter([self.X_next[0]],[self.X_next[1]],color = 'k',marker = '*')

            prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad = -10)

            plt.subplot(212)
            plt.contourf(self.xx, self.yy, np.reshape(ent, self.xx.shape))
            plt.colorbar(label = 'active learning [-]', ticks = [])
            plt.scatter(self.X[:,0],self.X[:,1],c=self.Y)


            if self.X_next is not None:
                plt.scatter([self.X_next[0]],[self.X_next[1]],color = 'k',marker = '*')

            prettyplot("$\mathregular{X_1}$", "$\mathregular{X_2}$", ylabelpad = -10)


        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)

    def append_next_point(self):
        self.X = np.vstack((self.X, self.X_next))
        Y_next = 1.0*self.sampling_func(denormalize(self.X_next[None,:],self.lb, self.ub))
        self.Y = np.concatenate((self.Y, np.array(Y_next)))


class GPclassifier(BaseSFclassifier):
    def create_model(self):
        with pm.Model() as self.model:
            self.mean = pm.gp.mean.Zero()

            # covariance function
            l = pm.Gamma("l_L", alpha=2, beta=2, shape = self.dim)
            # informative, positive normal prior on the period
            eta = pm.HalfNormal("eta_L", sd=5)
            self.cov = eta * pm.gp.cov.ExpQuad(self.dim, l)


            K = self.cov(self.X)

            self.K_stable = pm.gp.util.stabilize(K)


            v = pm.Normal("fp_rotated_", mu=0.0, sd=1.0, shape=pm.gp.util.infer_shape(self.X))
            fp = pm.Deterministic("fp", self.mean(self.X) + tt.slinalg.cholesky(self.K_stable).dot(v))

            p = pm.Deterministic("p", pm.math.invlogit(fp))
            y = pm.Bernoulli("y", p=p, observed=self.Y)

    def generate_samples(self,name, X_new, n_samples = 500):
        with self.model as model:

            Kxs = self.cov(self.X,X_new)

            L = tt.slinalg.cholesky(self.K_stable)
            A = pm.gp.util.solve_lower(L, Kxs)
            v2 = pm.gp.util.solve_lower(L, model.fp - self.mean(self.X))
            mu_pred = self.mean(X_new) + tt.dot(tt.transpose(A), v2)
            Kss = self.cov(X_new) + 1e-6 * tt.eye(X_new.shape[0])
            cov_pred = Kss - tt.dot(tt.transpose(A), A)
            f_pred = pm.MvNormal(name, mu=mu_pred, cov=cov_pred, shape=pm.gp.util.infer_shape(X_new))

        with self.model:
            pred_samples = pm.sample_ppc(self.trace, vars=[f_pred], samples=n_samples)

        return pred_samples

class SGPclassifier(BaseSFclassifier):
    def __init__(self,X, Y, Xu, lb, ub,sampling_func, X_test = None, Y_test = None, N_cand = 1000, boundary = None):
        super(SGPclassifier, self).__init__(X, Y, lb, ub,sampling_func, X_test = X_test, Y_test = Y_test, N_cand = 1000, boundary = None)
        self.Xu = normalize(Xu, lb, ub)
        self.sigma = 0.01 # jitter for u-f variance

    def create_model(self):
        with pm.Model() as self.model:
            self.mean = pm.gp.mean.Zero()

            # covariance function
            l = pm.Gamma("l_L", alpha=2, beta=2, shape = self.dim)
            # informative, positive normal prior on the period
            eta = pm.HalfNormal("eta_L", sd=5)
            self.cov = eta * pm.gp.cov.ExpQuad(self.dim, l)

            Kuu = self.cov(self.Xu)
            Kuf = self.cov(self.Xu, self.X)
            Luu = tt.slinalg.cholesky(pm.gp.util.stabilize(Kuu))

            vu = pm.Normal("u_rotated_", mu=0.0, sd=1.0, shape=pm.gp.util.infer_shape(self.Xu))
            u = pm.Deterministic("u", Luu.dot(vu))

            Luuinv_u = pm.gp.util.solve_lower(Luu,u)
            A = pm.gp.util.solve_lower(Luu, Kuf)

            Qff = tt.dot(tt.transpose(A),A)

            Kffd = self.cov(self.X, diag=True)
            Lamd = pm.gp.util.stabilize(tt.diag(tt.clip(Kffd - tt.diag(Qff) + self.sigma**2, 0.0, np.inf)))


            v = pm.Normal("fp_rotated_", mu=0.0, sd=1.0, shape=pm.gp.util.infer_shape(self.X))
            fp = pm.Deterministic("fp", tt.dot(tt.transpose(A), Luuinv_u) + tt.sqrt(Lamd).dot(v))

            p = pm.Deterministic("p", pm.math.invlogit(fp))
            y = pm.Bernoulli("y", p=p, observed=self.Y)

    def generate_samples(self,name, X_new, n_samples = 500):
        with self.model as model:

            Kuu = pm.gp.util.stabilize(self.cov(self.Xu))
            Kuf = self.cov(self.Xu, self.X)
            Luu = tt.slinalg.cholesky(Kuu)
            A = pm.gp.util.solve_lower(Luu, Kuf)
            Qff = tt.dot(tt.transpose(A),A)
            Kffd = self.cov(self.X, diag=True)
            Lamd_inv = tt.diag(1./tt.clip(Kffd - tt.diag(Qff) + self.sigma**2, 0, np.inf))

            Sigma = pm.gp.util.stabilize(Kuu + tt.dot(Kuf.dot(Lamd_inv),tt.transpose(Kuf)))
            L_Sigma = tt.slinalg.cholesky(Sigma)


            Kus = self.cov(self.Xu,X_new)

            m1 = pm.gp.util.solve_lower(L_Sigma, Kus)
            m2 = pm.gp.util.solve_lower(L_Sigma, Kuf)

            mu_pred = tt.dot(tt.dot(tt.transpose(m1),m2),tt.dot(Lamd_inv,model.fp))

            Kss = self.cov(X_new) + 1e-6 * tt.eye(X_new.shape[0])
            As = pm.gp.util.solve_lower(Luu, Kus)
            Qss = tt.dot(tt.transpose(As),As)


            cov_pred = Kss - Qss + tt.dot(tt.transpose(m1),m1)

            f_pred = pm.MvNormal(name, mu=mu_pred, cov=cov_pred, shape=pm.gp.util.infer_shape(X_new))

        with self.model:
            pred_samples = pm.sample_ppc(self.trace, vars=[f_pred], samples=n_samples)

        return pred_samples


class MFGPclassifier(BaseMFclassifier):

    def create_model(self):
        with pm.Model() as self.model:
            self.mean = pm.gp.mean.Zero()

            # covariance function
            l_L = pm.Gamma("l_L", alpha=2, beta=2, shape = self.dim)
            # informative, positive normal prior on the period
            eta_L = pm.HalfNormal("eta_L", sd=5)
            self.cov_L = eta_L * pm.gp.cov.ExpQuad(self.dim, l_L)

                # covariance function
            l_H = pm.Gamma("l_H", alpha=2, beta=2, shape = self.dim)
            delta = pm.Normal("delta", sd=10)
            # informative, positive normal prior on the period
            eta_H = pm.HalfNormal("eta_H", sd=5)
            self.cov_H = eta_H * pm.gp.cov.ExpQuad(self.dim, l_H)

            K_LL = self.cov_L(self.X_L)
            K_HH = delta**2*self.cov_L(self.X_H) + self.cov_H(self.X_H)
            K_LH = delta*self.cov_L(self.X_L, self.X_H)

            K1 = tt.concatenate([K_LL, K_LH], axis = 1)
            K2 = tt.concatenate([K_LH.T, K_HH], axis = 1)
            self.K_stable = pm.gp.util.stabilize(tt.concatenate([K1,K2], axis = 0))


            v = pm.Normal("fp_rotated_", mu=0.0, sd=1.0, shape=pm.gp.util.infer_shape(self.X))
            fp = pm.Deterministic("fp", self.mean(self.X) + tt.slinalg.cholesky(self.K_stable).dot(v))

            p = pm.Deterministic("p", pm.math.invlogit(fp))
            y = pm.Bernoulli("y", p=p, observed=self.Y)


    def generate_samples(self,name, X_new, n_samples = 500):
        with self.model as model:
            K_xs1 = model.delta*self.cov_L(X_new, self.X_L)
            K_xs2 = model.delta**2*self.cov_L(X_new, self.X_H) + self.cov_H(X_new, self.X_H)


            Kxs = tt.concatenate([K_xs1, K_xs2], axis = 1).T

            L = tt.slinalg.cholesky(self.K_stable)
            A = pm.gp.util.solve_lower(L, Kxs)
            v2 = pm.gp.util.solve_lower(L, model.fp - self.mean(self.X))
            mu_pred = self.mean(X_new) + tt.dot(tt.transpose(A), v2)
            Kss = model.delta**2*self.cov_L(X_new) + self.cov_H(X_new) + 1e-6 * tt.eye(X_new.shape[0])
            cov_pred = Kss - tt.dot(tt.transpose(A), A)
            f_pred = pm.MvNormal(name, mu=mu_pred, cov=cov_pred, shape=pm.gp.util.infer_shape(X_new))
            #f_pred = gp.conditional("f_pred", X_new)

        with self.model:
            pred_samples = pm.sample_ppc(self.trace, vars=[f_pred], samples=n_samples)

        return pred_samples


class SMFGPclassifier(BaseMFclassifier):
    def __init__(self,X_L, Y_L, X_H, Y_H, X_Lu, X_Hu, lb, ub, sampling_func, X_test = None, Y_test = None, N_cand = 1000, boundary_H = None, boundary_L = None):
        super(SMFGPclassifier, self).__init__(X_L, Y_L, X_H, Y_H, lb, ub,sampling_func, X_test = X_test, Y_test = Y_test, N_cand = 1000, boundary_H = None, boundary_L = None)

        self.X_Lu = normalize(X_Lu, lb, ub)
        self.X_Hu = normalize(X_Hu, lb, ub)
        self.Xu = np.concatenate((X_Lu,X_Hu))

        self.sigma = 0.1

    def create_model(self):
        with pm.Model() as self.model:
                # Again, f_sample is just a dummy variable
            self.mean = pm.gp.mean.Zero()

            # covariance function
            l_L = pm.Gamma("l_L", alpha=2, beta=2, shape = self.dim)
            # informative, positive normal prior on the period
            eta_L = pm.HalfNormal("eta_L", sd=5)
            self.cov_L = eta_L * pm.gp.cov.ExpQuad(self.dim, l_L)

                # covariance function
            l_H = pm.Gamma("l_H", alpha=2, beta=2, shape = self.dim)
            delta = pm.Normal("delta", sd=10)
            # informative, positive normal prior on the period
            eta_H = pm.HalfNormal("eta_H", sd=5)
            self.cov_H = eta_H * pm.gp.cov.ExpQuad(self.dim, l_H)

            ###############################################################
            #compute Kuu
            K_LLu = self.cov_L(self.X_Lu)
            K_HHu = delta**2*self.cov_L(self.X_Hu) + self.cov_H(self.X_Hu)
            K_LHu = delta*self.cov_L(self.X_Lu, self.X_Hu)

            K1 = tt.concatenate([K_LLu, K_LHu], axis = 1)
            K2 = tt.concatenate([K_LHu.T, K_HHu], axis = 1)
            self.Kuu= pm.gp.util.stabilize(tt.concatenate([K1,K2], axis = 0))


            ##############################################################
            #compute Kuf
            K_LLuf = self.cov_L(self.X_Lu, self.X_L) # uL x L
            K_HHuf = delta**2*self.cov_L(self.X_Hu, self.X_H) + self.cov_H(self.X_Hu, self.X_H) # uH x H
            K_LHuf = delta*self.cov_L(self.X_Lu, self.X_H) # uL x H
            K_HLuf = delta*self.cov_L(self.X_Hu, self.X_L) # uH x L

            K1 = tt.concatenate([K_LLuf, K_LHuf], axis = 1)
            K2 = tt.concatenate([K_HLuf, K_HHuf], axis = 1)
            self.Kuf= pm.gp.util.stabilize(tt.concatenate([K1,K2], axis = 0))

            ##############################################################

            self.Luu = tt.slinalg.cholesky(self.Kuu)


            vu = pm.Normal("u_rotated_", mu=0.0, sd=1.0, shape=pm.gp.util.infer_shape(self.Xu))
            u = pm.Deterministic("u", self.Luu.dot(vu))

            Luuinv_u = solve_lower_triangular(self.Luu,u)
            A = solve_lower_triangular(self.Luu, self.Kuf)

            self.Qffd = tt.sum(A * A, 0)

            K_LLff = self.cov_L(self.X_L, diag = True)
            K_HHff = delta**2*self.cov_L(self.X_Hu, diag = True) + self.cov_H(self.X_Hu, diag = True)

            Kffd = tt.concatenate([K_LLff, K_HHff])
            self.Lamd = tt.clip(Kffd - self.Qffd , 0.0, np.inf) + self.sigma**2

            v = pm.Normal("fp_rotated_", mu=0.0, sd=1.0, shape=pm.gp.util.infer_shape(self.X))
            fp = pm.Deterministic("fp", tt.dot(tt.transpose(A), Luuinv_u) + tt.sqrt(self.Lamd)*v)

            p = pm.Deterministic("p", pm.math.invlogit(fp))
            y = pm.Bernoulli("y", p=p, observed=self.Y)


    def generate_samples(self,name, X_new, n_samples = 500):
        with self.model as model:
            A = pm.gp.util.solve_lower(self.Luu, self.Kuf)
            A_l = A / self.Lamd
            L_B = tt.slinalg.cholesky(tt.eye(self.Xu.shape[0]) + tt.dot(A_l, tt.transpose(A)))
            r_l = model.fp / self.Lamd
            c = pm.gp.util.solve_lower(L_B, tt.dot(A, r_l))
            K_us1 = model.delta*self.cov_L(X_new, self.X_Lu)
            K_us2 = model.delta**2*self.cov_L(X_new, self.X_Hu) + self.cov_H(X_new, self.X_Hu)
            Kus = tt.concatenate([K_us1, K_us2], axis = 1).T
            As = pm.gp.util.solve_lower(self.Luu, Kus)
            mu_pred = tt.dot(tt.transpose(As), pm.gp.util.solve_upper(tt.transpose(L_B), c))
            C = pm.gp.util.solve_lower(L_B, As)
            Kss = model.delta**2*self.cov_L(X_new) + self.cov_H(X_new) + 1e-6 * tt.eye(X_new.shape[0])
            Qss = tt.dot(tt.transpose(As),As)
            cov_pred = (Kss - Qss + tt.dot(tt.transpose(C), C))
            f_pred = pm.MvNormal(name, mu=mu_pred, cov=cov_pred, shape=pm.gp.util.infer_shape(X_new))
            #f_pred = gp.conditional("f_pred", X_new)
        with self.model:
            pred_samples = pm.sample_ppc(self.trace, vars=[f_pred], samples=n_samples)
        return pred_samples








