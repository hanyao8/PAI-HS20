import GPy
import scipy
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

domain = np.array([[0, 5]])


""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        #pass
        self.X = np.empty(shape = (1,1)) 
        self.Y = np.empty(shape = (1,1))
        self.Y_time = np.empty(shape = (1,1))
        self.model_performance = None
        self.first_run = 1
        self.j_rec = 0
        self.j_add = 0
        self.af_type = "GPUCB"
        self.rec_warmup = 5
        self.gpucb_kappa = 1.0


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        #raise NotImplementedError
        if self.j_rec<self.rec_warmup:
            recommendation = np.array([np.ones((domain.shape[0]))])
            recommendation *= domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
        else: 
            recommendation = self.optimize_acquisition_function()

        print("optimize%d"%(self.j_rec))
        print(recommendation)
        self.j_rec += 1
        return recommendation


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        #raise NotImplementedError

        if self.af_type == "EI":
            #m,  s  = self.model_performance.predict(np.transpose(self.model_performance.X))
            x_acq = np.linspace(0,5,100).reshape((100,1))
            #m,  s  = self.model_performance.predict(self.model_performance.X)
            m,  s  = self.model_performance.predict(x_acq)
            #print("ms")
            #print(np.shape(m))
            self.mshape = np.shape(m)
            #print(np.shape(s))
            self.sshape = np.shape(s)
            self.predictive_mean = m[0][0]
            self.predictive_sigma = s[0][0] 
            
            z_x = ( x - self.predictive_mean ) / self.predictive_sigma 
            PHI = scipy.stats.norm.pdf(z_x) 
            phi = scipy.stats.norm.cdf(z_x) 
            acq_value = self.predictive_sigma  * (z_x * PHI + phi)

        elif self.af_type == "GPUCB":
            m,  s  = self.model_performance.predict(np.atleast_2d(x))
            acq_value =  m[0][0] + self.gpucb_kappa*s[0][0]
        return acq_value


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        # raise NotImplementedError
        if self.j_add==0:
            self.X = np.atleast_2d(x)
            self.Y = np.atleast_2d(f)
            self.Y_time = np.atleast_2d(v)
        else:
            self.X = np.concatenate((self.X, np.atleast_2d(x)), axis = 0) 
            self.Y = np.concatenate((self.Y,  np.atleast_2d(f)), axis = 0)
            self.Y_time = np.concatenate((self.Y_time ,  np.atleast_2d(v)), axis = 0)
        #print(np.atleast_2d(x))
        #print(self.Y.flatten())
        #print(self.Y_time.flatten())
        #print(self.X.flatten())
        
        if self.model_performance is None: 
            # TODO change smoothness
            #kern_perf = GPy.kern.Matern52(input_dim = len(self.X), variance=0.5, lengthscale=0.5) # SMOOTHNESS, period=2.5)
            kern_perf = GPy.kern.Matern52(input_dim = 1, variance=0.5, lengthscale=0.5) # SMOOTHNESS, period=2.5) 
            self.model_performance = GPy.models.GPRegression(self.X,   self.Y , kernel=kern_perf, noise_var=0.15)
            #kern_speed = GPy.kern.Matern52(input_dim = len(x), variance=0.5, lengthscale=0.5) # SMOOTHNESS, period=2.5) 
            #self.model_speed= GPy.models.GPRegression(x, np.atleast_2d(f), kernel=kern_speed, noise_var=0.15)
     
        else:
            self.model_performance.set_XY(self.X , self.Y ) # set is not additive but replaces
        self.j_add += 1

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        # raise NotImplementedError
        print(self.Y.flatten())
        print(self.Y_time.flatten())
        print(self.X.flatten())
        idx = np.argmax(np.array(self.Y))
        solution = self.X[idx,0]
        print(solution)
        return solution

""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    f = - (np.linalg.norm(x - mid_point, 2))**2
    f += np.random.normal(scale=0.15)
    return f  # -(x - 2.5)^2
    #to maximize
    #sqrt(sq(diff))


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()


    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    x_plot = np.linspace(0,5,100)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()
        print(x)

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)
        ax1.plot(x_plot,
                agent.model_performance.predict(x_plot.reshape((100,1)))[0].flatten(),
                alpha=j/20)
        if j==19:

            ax1.plot(
                    x_plot,
                    agent.model_performance.predict(x_plot.reshape((100,1)))[0].flatten(),
                    linewidth=4
                    )
            ax1.plot(agent.X.flatten()[-10:],agent.Y.flatten()[-10:],linewidth=1.0)


    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')
    return(agent)


if __name__ == "__main__":
    agent = main()

    plt.show()
