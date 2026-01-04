from src.dMALIBOO.bayesian_optimization import BayesianOptimization as BO
import numpy as np
import os


class BOmanager(object):
    """
    Manager class for running Bayesian Optimization using the D-MALIBOO approach.
    """

    def __init__(self, f, pbounds, dataset, target_column, constraint_column, seed, output_path, init_points=None, n_iter=1, log_name='results', decay_bounds=None):
        """
        Initialize the Bayesian Optimization manager.

        Parameters:
            :param function f               : function to optimize
            :param dict pbounds             : dictionary of parameter bounds
            :param str dataset              : path to dataset
            :param str target_column        : target column to optimize
            :param str constraint_column    : constraint column to constrain
            :param int seed                 : random seed for reproducibility
            :param str output_path          : path to output directory
            :param iterable init_points     : initial points for optimization
            :param int n_iter               : number of iterations for optimization
            :param str log_name             : name of log file
            :param decay_bounds             : decay bounds for optimization
        """
        output_path = output_path + '/BO_logs/'

        if os.path.exists(f"{output_path}/{log_name}"):
            os.remove(f"{output_path}/{log_name}")

        optimizer = BO(f=f, 
                       pbounds=pbounds,
                       dataset=dataset, 
                       target_column=target_column,
                       random_state=seed, 
                       output_path=output_path, 
                       verbose=0,
                       log_name=log_name,
                       decay_bounds=decay_bounds)
    
        optimizer.add_initial_points(init_points)

        acq_info = {'DBO_kernel': 'Matern', 'initial_nu': 2.5, 'nu_h': 10.0,
                    'ml_target': constraint_column, 'ml_bounds': (0.0, 2.1), 'ml_bounds_type': 'indicator', 'ml_bounds_model': 'Ridge', 'ml_bounds_alpha': 0.099, 
                    'ml_target_type': 'probability', 'ml_target_model': 'Ridge', 'ml_target_coeff': (1.111, None), 'ml_target_alpha': 0.959, 
                    'eps_greedy_random_prob': 0.1}
    
        optimizer.maximize(init_points = 0, initial_points_selection_method = 'random',
                           n_iter = n_iter, 
                           acq = 'ei',
                           ml_on_bounds = 'indicator', 
                           ml_on_target = 'probability',
                           epsilon_greedy = 'True',
                           adaptive_method = 'None',
                           memory_queue_len = len(init_points) + n_iter, 
                           acq_info = acq_info,
                           consider_max_only_on_feasible = 'True')
        
        self.all_points = optimizer._space._params
        self.points = optimizer._space._params.tail(-len(init_points))
        self.target = optimizer._space._target[len(init_points):]
        self.constr = np.array([acq_info['ml_bounds'][1] - _ for _ in optimizer._space._constr_values[len(init_points):]])
        self.feasibility = optimizer._space._feasibility[len(init_points):]
        self.best_solution = optimizer.max['params']
        self.best_target = -optimizer.max['target']
        self.best_feasible = optimizer.max['feasible']
        self.best_constraint = acq_info['ml_bounds'][1] - optimizer.max['constraint']


if __name__ == "__main__":
    # For testing purposes
    init_points = (
        {'ALIGN_SPLIT': 8, 'OPTIMIZE_SPLIT': 8, 'OPTIMIZE_REPS': 1, 'CUDA_THREADS': 128, 'N_RESTART': 1024, 'CLIPPING': 256, 'SIM_THRESH': 1, 'BUFFER_SIZE': 20},
        {'ALIGN_SPLIT': 32, 'OPTIMIZE_SPLIT': 12, 'OPTIMIZE_REPS': 2, 'CUDA_THREADS': 192, 'N_RESTART': 256, 'CLIPPING': 10, 'SIM_THRESH': 4, 'BUFFER_SIZE': 10},
        {'ALIGN_SPLIT': 72, 'OPTIMIZE_SPLIT': 12, 'OPTIMIZE_REPS': 1, 'CUDA_THREADS': 256, 'N_RESTART': 256, 'CLIPPING': 30, 'SIM_THRESH': 4, 'BUFFER_SIZE': 50},
        {'ALIGN_SPLIT': 72, 'OPTIMIZE_SPLIT': 48, 'OPTIMIZE_REPS': 3, 'CUDA_THREADS': 224, 'N_RESTART': 256, 'CLIPPING': 30, 'SIM_THRESH': 2, 'BUFFER_SIZE': 2},
        {'ALIGN_SPLIT': 20, 'OPTIMIZE_SPLIT': 48, 'OPTIMIZE_REPS': 1, 'CUDA_THREADS': 64, 'N_RESTART': 256, 'CLIPPING': 10, 'SIM_THRESH': 1, 'BUFFER_SIZE': 20},
        {'ALIGN_SPLIT': 16, 'OPTIMIZE_SPLIT': 32, 'OPTIMIZE_REPS': 2, 'CUDA_THREADS': 128, 'N_RESTART': 1024, 'CLIPPING': 10, 'SIM_THRESH': 3, 'BUFFER_SIZE': 1},
        {'ALIGN_SPLIT': 12, 'OPTIMIZE_SPLIT': 8, 'OPTIMIZE_REPS': 2, 'CUDA_THREADS': 160, 'N_RESTART': 256, 'CLIPPING': 30, 'SIM_THRESH': 4, 'BUFFER_SIZE': 2},
        {'ALIGN_SPLIT': 16, 'OPTIMIZE_SPLIT': 24, 'OPTIMIZE_REPS': 1, 'CUDA_THREADS': 128, 'N_RESTART': 1024, 'CLIPPING': 256, 'SIM_THRESH': 1, 'BUFFER_SIZE': 20},
        {'ALIGN_SPLIT': 32, 'OPTIMIZE_SPLIT': 20, 'OPTIMIZE_REPS': 1, 'CUDA_THREADS': 224, 'N_RESTART': 256, 'CLIPPING': 50, 'SIM_THRESH': 1, 'BUFFER_SIZE': 2},
        {'ALIGN_SPLIT': 20, 'OPTIMIZE_SPLIT': 72, 'OPTIMIZE_REPS': 3, 'CUDA_THREADS': 256, 'N_RESTART': 1024, 'CLIPPING': 30, 'SIM_THRESH': 3, 'BUFFER_SIZE': 20}
    )

    BOmanager(
        f = None,
        pbounds = {'ALIGN_SPLIT': (8, 72.01), 'OPTIMIZE_SPLIT': (8, 72.01), 'OPTIMIZE_REPS': (1, 5.01), 'CUDA_THREADS': (32, 256.01), 'N_RESTART': (256, 1024.01), 'CLIPPING': (10, 256.01), 'SIM_THRESH': (1, 4.01), 'BUFFER_SIZE': (1, 50.01)},
        dataset = "ligen_synth_table.csv",
        target_column = 'RMSD^3*TIME',
        constraint_column = 'RMSD_0.75',
        seed = 1,
        output_path = "outputs/test",
        init_points = init_points
    )