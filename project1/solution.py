import numpy as np

import pandas as pd
import scipy.integrate as integrate
from scipy.stats import norm as gaussian

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel,RBF,Matern

import kernels
import utils

import os
import logging
import sys

THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.01


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)


"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""



class Model():
    def __init__(self, model_config_override={}):
        

        """
            TODO: enter your code here
        """
        model_config = {
                "use_skit_learn":False, 
                "use_nystrom":True,
                "nystrom_q":100,
                "kernel":kernels.sklearn_best(),
                "variance":1,
                "correct_y_pred":False,
                "model_preprocess_left_frac":0.75 }
        for k in list(model_config_override.keys()):
            model_config[k] = model_config_override[k]

        self.rbf_w = 1.0
        self.rbf_ls = np.exp(-1.18020928)
        self.wk_nl = np.exp(-5.85276903)

        self.kernel = model_config["kernel"]
        #self.THRESHOLD = THRESHOLD
        self.variance = model_config["variance"]
        self.use_skit_learn = model_config["use_skit_learn"]
        self.use_nystrom = model_config["use_nystrom"]
        self.q = model_config["nystrom_q"]

        self.correct_y_pred = model_config["correct_y_pred"]
        #self.public_score_run = public_score_run
        self.model_preprocess_left_frac = model_config["model_preprocess_left_frac"]

        #"implicit" attributes:
        self.rho = None
        self.correction_obj = None
        self.correction_obj_vals_sample = None
        self.test_x = None
        pass

    def model_preprocess(self,train_x,train_y):
        df_vals = np.stack([train_x[:,0],train_x[:,1],train_y],axis=1)
        print(df_vals)
        print(df_vals.shape)
        df = pd.DataFrame(data = df_vals,columns = ['x0','x1','y'])
        print(df.shape)
        self.df = df 
        
        df_left = df[df['x0']<-0.5]
        df_left = df_left.sample(frac=self.model_preprocess_left_frac,random_state=42)
        df_right = df[df['x0']>-0.5]
        self.df_left = df_left
        self.df_right = df_right

        self.df_left_right = pd.concat([self.df_left,self.df_right])

        train_x = self.df_left_right[['x0','x1']].values
        train_y = self.df_left_right['y'].values
        return train_x,train_y

    def correction_obj_integrand(self,y1,y2):
        return cost_function(np.array([y1]),np.array([y2]))*self.rho(y1)
    
    def correction_obj(self,y2,y1_bounds):
        res = integrate.quad(
                lambda y1: self.correction_obj_integrand(y1,y2),
                -np.inf,np.inf)
        return res[0]

    def correct_y(self,y_mean,y_std):
        self.rho = gaussian(loc=y_mean,scale=y_std).pdf
        test_bounds = [y_mean-2*y_std,y_mean+2*y_std]
        test_points = np.linspace(test_bounds[0],test_bounds[1],100)
        correction_obj_vals = np.array([self.correction_obj(tp,test_bounds)\
                for tp in test_points])
        self.correction_obj_vals_sample = correction_obj_vals
        y_mean_corrected = test_points[np.argmin(correction_obj_vals)]
        return y_mean_corrected


    def predict(self, test_x):
        """
            Uses either our implementation our the skitlearn model
            TODO: enter your code here
        """
        self.test_x = test_x  
        
        if self.use_skit_learn:
            y = self.fitted.predict(self.test_x)
        else:
            K_Q_x = self.kernel(self.test_x, self.train_x)
            K_x_Q = self.kernel(self.train_x, self.test_x)
            K_Q_Q = self.kernel(self.test_x, self.test_x)
            means = K_Q_x.dot(self.K_x_x_inv).dot(self.train_y)
            cov = K_Q_Q - K_Q_x.dot(self.K_x_x_inv).dot(K_x_Q)
            cov = (cov+cov.transpose())/2
            vars_val = np.diag(cov)
            print(means)
            print(cov)
            y = (np.random.multivariate_normal(means.ravel(), cov, 1)).flatten() #sample from the multivar normal
            print(y.shape)
            logging.info(str(y.shape))
            print(y)

        if self.correct_y_pred:
        #y_correction vectorization in dev
            y_mean_corrected = np.array([])
            for i in range(0,len(self.test_x)):
                y_mean_corrected = np.append(y_mean_corrected,self.correct_y(y_mean,y_std))
            return y_mean_corrected

        y = np.clip(y,a_min=0,a_max=1)
        y = np.sqrt(y)
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        logging_setup()

        if self.model_preprocess_left_frac<0.99:
            self.train_x,self.train_y = self.model_preprocess(train_x,train_y)
        else:
            self.train_x = train_x
            self.train_y = train_y
        
        self.n = np.shape(self.train_x)[0]
        if self.use_skit_learn:
            self.gpr = GaussianProcessRegressor(kernel=self.kernel,copy_X_train=False,random_state=42)
            self.fitted = self.gpr.fit( self.train_x , self.train_y)
        else:
            if self.use_nystrom:
                logging.info("Using Nystrom")
                K_nq = self.kernel(self.train_x,self.train_x[:self.q])
                K_qq = self.kernel(self.train_x[:self.q],self.train_x[:self.q])
                K_qq_eigendec = np.linalg.eig(K_qq)
                Lambda_qq = np.diag(K_qq_eigendec[0])
                U_qq = K_qq_eigendec[1]

                #For the minor components- U_qq contains some complex columns
                #if we don't do PCA
                p = utils.pca_find_p(K_qq_eigendec[0],pca_thresh=(1-1e-2))
                self.K_qq_eigendec = K_qq_eigendec
                self.p = p
                Lambda_pp = np.diag(K_qq_eigendec[0][:p+1])
                U_qp = U_qq[:,:p+1]

                Q = K_nq.dot(U_qp).dot(np.sqrt(np.linalg.inv(Lambda_pp)))
                self.U_qp = U_qp
                self.K_nq = K_nq
                self.Lambda_pp = Lambda_pp

                self.Q = Q

                #reduced_inv = np.linalg.inv( self.variance*np.eye(self.q)+(Q.transpose()).dot(Q) )
                reduced_inv = np.linalg.inv( self.variance*np.eye(Q.shape[1])+(Q.transpose()).dot(Q) )
                self.K_x_x_inv = (1/self.variance)*np.eye(self.n) - \
                        (1/self.variance)*Q.dot(reduced_inv).dot(Q.transpose()).real
            else:
                self.K_x_x = self.kernel(self.train_x, self.train_x)
                self.K_x_x_inv = np.linalg.inv(self.K_x_x +
                                                   (self.variance * np.eye(*self.K_x_x.shape)))
        
    def likelihood(self):
        if self.use_skit_learn:
            log_likelihood = log_marginal_likelihood()
        else:
            log_likelihood = (-0.5 * np.dot(self.y_train.T, np.dot(self.K_train_train_inv, self.y_train))
                     -0.5 * np.log(np.linalg.det(self.K_train_train+np.eye(*self.K_train_train.shape)))
                     -0.5 * self.y_train.shape[0] * np.log(2*np.pi))
        return log_likelihood


class cv_eval():
    def __init__(self,
            cv_splits,
            cv_preprocess_left_frac,
            model_config):
        self.K_cv = cv_splits
        #self.model = Model(use_skit_learn, kernel=kernel, preprocess_frac=1.0)
        self.model = Model(model_config)
        self.cv_preprocess_left_frac = cv_preprocess_left_frac #run speed O(N^3)? Can set 0.01 for testing purposes
        self.model_config = model_config
    
    def preprocess(self, train_x, train_y):
        df_vals = np.stack([train_x[:,0],train_x[:,1],train_y],axis=1)
        print(df_vals)
        print(df_vals.shape)
        df = pd.DataFrame(data = df_vals,columns = ['x0','x1','y'])
        print(df.shape)
        self.df = df 
        
        df_left = df[df['x0']<-0.5]
        df_left = df_left.sample(frac=self.cv_preprocess_left_frac,random_state=42)
        df_right = df[df['x0']>-0.5]
        self.df_left = df_left
        self.df_right = df_right
    
    def get_split(self, i):
        ''' i index of the CV split'''
        print("i=%d"%i)
        df_train_list = [self.df_left]
        df_right_shuffle = self.df_right.sample(frac=1,random_state=42)
        df_right_splits = np.array_split(df_right_shuffle,self.K_cv)
        for j in range(0,self.K_cv):
            if j!=i:
                df_train_list.append(df_right_splits[j])
        #print(len(df_train_list))
        df_train = pd.concat(df_train_list)
        df_val = df_right_splits[i]

        self.X_train = df_train[['x0','x1']].values
        self.y_train = df_train['y'].values
        self.X_val = df_val[['x0','x1']].values
        self.y_val = df_val['y'].values
    
    def run_cross_validation(self, train_x, train_y):
        if self.cv_preprocess_left_frac < 0.999:
            self.preprocess(train_x, train_y)
        val_cost_array = np.array([])
        self.models = []
        for i in range(0,self.K_cv):
            self.get_split(i)
            self.model.fit_model(self.X_train, self.y_train)
            if self.model_config["use_skit_learn"]:
                self.models.append(self.model.fitted)
           # print("training gpr score %f"%(gpr.score(X_train,y_train)))

            y_train_pred = self.model.predict(self.X_train)
            #print(np.shape(y_train_pred))
            print("training cost fn   %f"%(cost_function(self.y_train,y_train_pred)))
            
            y_val_pred = self.model.predict(self.X_val)
            #print(y_val_pred)
            #print(np.shape(y_val_pred))
            
            val_cost = cost_function(self.y_val,y_val_pred)
            print("val cost fn        %f"%(val_cost))
            val_cost_array = np.append(val_cost_array,val_cost)
            print("\n")
        return val_cost_array



def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)

def logging_setup():
    print("logging_setup")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    logging_path = os.path.join(*[project_dir,'train.log'])
    logging.basicConfig(filename=logging_path, filemode='a',\
            format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info("Begin logging for single training run")
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

if __name__ == "__main__":
    main()
