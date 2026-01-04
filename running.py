# simple entrypoint for running a local Hybrid Predictive Coding experiment.
# It imports the default configuration, overrides a few settings (log directory, which
# evaluation modes to use, dataset sizes, and the number/threshold of inference iterations),
# and then calls the main training/evaluation loop from `pybrid.scripts.main(cfg)`.
from pybrid.scripts import main
from pybrid.config import default_cfg as cfg 
import time


'---------------------hybrid predictive coding trening------------------------- '
cfg.exp.log_dir = "results/local_hybrid/"
# cfg.exp.test_hybrid = True
# cfg.exp.test_amort = True

# cfg.data.train_size = None  
# cfg.data.test_size  = None  

start_time = time.time()
main(cfg)
run_time = time.time() - start_time
print('Evaluate time:', run_time)


'-------------------only local precoding training-----------------------------------'
# cfg.exp.log_dir = "results/local_predcoding"
# cfg.exp.test_pc = True
# cfg.model.train_amort = False

# cfg.data.train_size = 2000
# cfg.data.test_size = 1000

# cfg.infer.num_train_iters = 20
# cfg.infer.num_test_iters = 100

# cfg.infer.train_thresh = 0.001
# cfg.infer.test_thresh = 0.001

# main(cfg)


'-------------------some kind of splits training--------------------------------'
# from pybrid.split_scripts import main
# from pybrid.config import default_cfg as cfg

# cfg.exp.log_dir = "results/hybrid_split"
# cfg.exp.num_batches = 3000
# cfg.exp.batches_per_epoch = 100
# cfg.exp.test_hybrid = True
# cfg.exp.test_amort = True

# cfg.data.train_size = None
# cfg.data.test_size = None

# cfg.infer.num_train_iters = 100
# cfg.infer.num_test_iters = 100

# cfg.infer.train_thresh = 0.005
# cfg.infer.test_thresh = 0.005

# seeds = [0, 1, 2]
# for seed in seeds:
#     cfg.exp.seed = seed
#     main(cfg)