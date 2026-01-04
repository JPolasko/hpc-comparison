#defines the default experiment configuration used throughout the project. 

from pybrid import utils

#dosiahne 85 accuracy juchuuuu
default_cfg = {
    "exp": {
        "log_dir": "results/test",
        "seed": 0,
        "num_epochs": 30,
        "num_batches": 3000,
        "batches_per_epoch": 110,

        "test_every": 1,
        "test_hybrid": True,
        "test_pc": True,
        "test_amort": True,

        "log_batch_every": 100,
        "switch": None,
    },

    "data": {
        "train_size": None,   # full
        "test_size": None,    # full
        "label_scale": 0.94,
        "normalize": True,
    },

    "infer": {
        "mu_dt": 0.01,         # paper 0.01

        "num_train_iters": 300,         # paper default is 100, chcem ale lepsi trening
        "num_test_iters": 300,

        "fixed_preds_train": False,
        "fixed_preds_test": False,

        "train_thresh": 0.003,         # paper 0.05 aby iters klesali po case
        "test_thresh": 0.003,

        "init_std": 0.005,         

        "no_backward": False,
    },

    "model": {
        "nodes": [10, 500, 500, 784],
        "amort_nodes": [784, 500, 500, 10],

        "train_amort": True,
        "use_bias": True,

        "kaiming_init": False,     #True iba ak nie tanh
        "act_fn": "tanh",
    },

    "optim": {
        "name": "Adam",

        "lr": 1e-4,         # paper  1e-2 
        "amort_lr": 1e-4,

        "batch_size": 64,
        "test_batch_size": 64,

        "batch_scale": True,

        "grad_clip": 30,         # paper 50

        "weight_decay": 1e-4, #??

        "normalize_weights": True,         # based on paper
    },
}

default_cfg = utils.to_attr_dict(default_cfg)