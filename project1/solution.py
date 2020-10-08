import numpy as np

import pandas as pd
import scipy.integrate as integrate
from scipy.stats import norm as gaussian

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel,RBF,Matern

import kernels

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
    cost = (true - predicted) ** 2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted > THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2
    cost[mask_w3] = cost[mask_w3] * W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted < true, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2

    # reward for correctly identified safe regions
    reward = W4 * np.logical_and(predicted <= THRESHOLD, true <= THRESHOLD)

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
    def __init__(self, use_skit_learn=True, kernel=kernels.custom_kernel1, variance=1, correct_y_pred=False):
        """
            TODO: enter your code here
        """
        self.rbf_w = 1.0
        self.rbf_ls = np.exp(-1.18020928)
        self.wk_nl = np.exp(-5.85276903)

        self.kernel = kernel
        #self.THRESHOLD = THRESHOLD
        self.variance = variance
        self.use_skit_learn = use_skit_learn
        self.correct_y_pred = correct_y_pred

        #"implicit" attributes:
        self.rho = None
        self.correction_obj = None
        self.correction_obj_vals_sample = None
        self.test_x = None
        pass

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
            means = np.dot(K_Q_x, np.dot(self.K_x_x_inv, self.train_y))
            cov = K_Q_Q - np.dot(K_Q_x, np.dot(self.K_x_x_inv, K_x_Q))
            vars_val = np.diag(cov)
            y = np.random.multivariate_normal(means.ravel(), cov, 1) #sample from the multivar normal

        if self.correct_y_pred:
        #y_correction vectorization in dev
            y_mean_corrected = np.array([])
            for i in range(0,len(self.test_x)):
                y_mean_corrected = np.append(y_mean_corrected,self.correct_y(y_mean,y_std))
            return y_mean_corrected

        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        self.train_x = train_x
        self.train_y = train_y
        
        if self.use_skit_learn:
            self.gpr = GaussianProcessRegressor(kernel=self.kernel,copy_X_train=False,random_state=42)
            self.fitted = self.gpr.fit( self.train_x , self.train_y)
        
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
    def __init__(self, cv_splits, kernel, use_skit_learn=True):
        self.K_cv = cv_splits
        self.model = Model(use_skit_learn, kernel=kernel)
    
    def preprocess(self, train_x, train_y):
        df_vals = np.stack([train_x[:,0],train_x[:,1],train_y],axis=1)
        print(df_vals)
        print(df_vals.shape)
        df = pd.DataFrame(data = df_vals,columns = ['x0','x1','y'])
        print(df.shape)
        self.df = df 
        
        df_left = df[df['x0']<-0.5]
        df_left = df_left.sample(frac=0.1,random_state=42)
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
        self.preprocess(train_x, train_y)
        val_cost_array = np.array([])
        models = []
        for i in range(0,self.K_cv):
            self.get_split(i)
            gpr = self.model.fit_model(self.X_train, self.y_train)
            #models.append(gpr)
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


if __name__ == "__main__":
    main()
