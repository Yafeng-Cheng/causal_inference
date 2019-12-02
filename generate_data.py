import numpy as np
import pandas as pd

class GenerateDate:
    """
    Generate data for causal inference simulation. The potential data generation process could be one of:
    - Y ~ tau*A + beta_x*X + beta_z*Z; tau~N(tau_0, signma_tau); no confounding
    - Y ~ tau*A + beta_x*X + beta_z*Z; tau~N(f(X, Z), signma_tau); no confounding
    - Y ~ tau*A + beta_x*X + beta_z*Z; tau~N(tau_0, signma_tau); A~X+Z
    - Y ~ tau*A + beta_x*X + beta_z*Z; tau~N(f(X, Z), signma_tau); A~X+Z
    
    """
    
    def __init__(self, sample_size = 500, hetero = False, confound = False, seed = 1):
        self.sample_size = sample_size
        self.confound = confound
        self.hetero = hetero
        
        np.random.seed(seed)
        
        
    
    def continuous_variable(self, theta_1 = [0,0,0], theta_2 = [1,1,1], dist = ['n','n','u']):
        """
        Generate continuous independent variables. The distrubution could be normal or uniform.
        
        ----
        Args:
            theta_1: list of the first parameter. For normal distribution, this is the mean. For uniform distribution, this is the lower bound. Note that length of the list is the number of variables to generate. 
            theta_2: list of the second parameter. For normal distribution, this is the variance. For uniform distribution, this is the upper bound.
            dist: list of the distributions. 'n' is for normal, and 'u' is for uniform.
        
        ---
        Return:
        
        """
        dist_dict = {
            'n':np.random.normal,
            'u':np.random.uniform
        }
        dist_fun = [dist_dict[dist_i] for dist_i in dist]
        
        X = np.concatenate([
            dist_fun[i](theta_1[i], theta_2[i], size = self.sample_size).reshape((-1,1)) for i in range(len(theta_1))
        ], axis = 1)
        
        self.X = X
        self.outcome_beta_x = np.random.normal(0,0.3,size = len(theta_1))
        self.assignment_beta_x = np.random.normal(0,0.3,size = len(theta_1))

        self.x_name = ['x_'+str(i) for i in range(len(theta_1))]
        
    def discrete_variable(self, p = [[0.5,0.5],[0.1,0.2,0.3],[0.8,0.2]]):
        """
        Generate discrete independent variables. The distribution is categorical for each sample with given probability.
        
        ---- 
        Args:
            p: list of list of probabilities. Number of items in p means the number of variables to generate. Number of elements in each item means the number of category of the distribution. 
        
        """
        p = [[p_i_i/sum(p_i) for p_i_i in p_i] for p_i in p]
        
        Z = [
            np.random.choice(len(p_i),size = self.sample_size, p = p_i).reshape((-1,1)) for p_i in p
        ]
        
        Z = [np.concatenate([
            (Z[z_i] == i).astype(int) for i in range(len(p[z_i])) if i != p[z_i].index(max(p[z_i]))
        ], axis = 1) for z_i in range(len(Z))]
        
        self.Z = np.concatenate(Z, axis = 1)
        
        self.outcome_beta_z = [np.random.normal(0,0.3,size = len(self.Z[0]))]
        self.assignment_beta_z = [np.random.normal(0,0.3,size = len(self.Z[0]))]

        z_name = [[
            'x{}_{}'.format(str(z_i),str(i)) for i in range(len(p[z_i])) if i != p[z_i].index(max(p[z_i]))
        ] for z_i in range(len(Z))]
        
        self.z_name = sum(z_name, [])
    
    
    def defaul_model(
        self,
        beta_0 = None,
        beta_continoues=None, beta_categorical=None, 
        continoues = True, categorical = True
    ):
        if beta_0 is None:
            beta_0 = np.random.normal(0,0.3, size = 1)
        link_ = np.repeat(beta_0, self.sample_size)
        if continoues:
            if beta_continoues is None:
                beta_continoues = np.zeros(len(self.X[0]))
            link_ = link_+np.matmul(self.X,beta_continoues)
        if categorical:
            if beta_categorical is None:
                beta_categorical = np.zeros(len(self.Z[0]))
            link_ = link_+np.matmul(self.Z,beta_categorical)
        
        return link_
    
    
    def generate_A(
        self, assignment_model = None,
        beta_0 = None, 
        beta_continoues=None, beta_categorical=None, 
        continoues = True, categorical = True
    ):
        if self.confound:
            if assignment_model is None:
                assignment_model = self.defaul_model
            else:
                pass
            
            assignment_link = assignment_model(
                beta_0 = None,
                beta_continoues=beta_continoues, beta_categorical=beta_categorical, 
                continoues = continoues, categorical = categorical
            )
            assignment_p = self.my_logit(assignment_link)
            
            self.A = np.array([
                np.random.choice([0,1], p=[
                    1-assignment_p[i], assignment_p[i]
                ], size=1)[0] for i in range(self.sample_size)
            ])
        else:
            self.A = np.random.choice([0,1], p=[0.5,0.5], size=self.sample_size)
    
    def default_tau_fun(self, x):
        return 1/(1+exp(-20*x+3))
    
    def my_logit(self, x_beta=None, p=None):
        """
        logit transformation. x and p cannot both be None.
        
        ----
        Args:
            x_beta: x*beta. one dimensional.
            p: probability to be 1
            
        ----
        Return:
            1/(1+exp(-x)) if x_beta is not None;
            log(p/(1-p)) if x_beta is None and p is not None
        """
        if x_beta is not None:
            return 1/(1+np.exp(-1*x_beta))
        elif p is not None:
            return np.log(p/(1-p))
        else:
            pass
            
    
    def generate_tau(
        self, tau_0 = None, tau_fun = None,
        continoues = True, categorical = True
    ):
        tau_basic = np.random.normal(tau_0, 0.3, size = self.sample_size)
        self.tau = np.zeros(self.sample_size)
        
        if self.hetero:
            if tau_fun is None:
                tau_fun = self.default_tau_fun
            else:
                pass
            if continoues:
                self.tau = self.tau + np.apply_along_axis(tau_fun, 0, self.X).sum(axis = 1)
            if categorical:
                self.tau = self.tau + np.apply_along_axis(tau_fun, 0, self.Z).sum(axis = 1)
        else:
            self.tau = tau_basic
        
        
    def generate_link_y(
        self, 
        beta_0 = None, 
        beta_continoues=None, beta_categorical=None, 
        continoues = True, categorical = True
    ):
        if beta_0 is None:
            beta_0 = np.random.normal(0, 0.3, size = 1)
        link_Y = np.repeat(beta_0, self.sample_size)
        if continoues:
            if beta_continoues is None:
                beta_continoues = self.beta_x
            link_Y = link_Y+np.matmul(self.X,beta_continoues)
        if categorical:
            if beta_categorical is None:
                beta_categorical = self.beta_z
            link_Y = link_Y+np.matmul(self.Z,beta_categorical)
        self.link_Y = link_Y
        
    def combine_components(
        self, link_fn=None,
        variable_generating_dict = None,
        tau_generating_dict = None,
        A_generating_dict = None,
        link_y_dict = None
    ):
        
        pass
    