from itertools import product
from pybrid.scripts import main
from pybrid.config import default_cfg as cfg #mine
from pybrid import datasets, utils
from collections.abc import Mapping
import json
from standart_networks.train import run_training, load_model_eval
from standart_networks.datasets import load_data  
from typing import Dict
import torch
import torch.nn.functional as F
from pybrid.models.hybrid import HybridModel
from pybrid import datasets
from torch.utils.data import DataLoader
'---------------hladanie optimalnych parametrov------------------------'
def parameter_search_hpc():
    mu_dts = [0.02, 0.01, 0.005, 0.002]
    train_iters = [10, 20, 50, 100, 150]
    test_iters = [10, 20, 50, 100, 200]
    test_threshes = [None, 0.01, 0.005, 0.002, 0.001]
    init_stds = [0.005, 0.01, 0.02]
    act_fns = ["tanh", "relu"]
    lrs = [1e-2, 3e-3, 1e-3]


    cfg_base = cfg
    i = 0
    for (act_fn, lr, mu_dt, ntr, nte, tth) in product(act_fns, lrs, mu_dts, train_iters, test_iters, test_threshes ):
        cfg_run = utils.to_attr_dict(
            {k: ( {kk: ( {kkk: vvv for kkk, vvv in vv.items()} if isinstance(vv, Mapping) else vv)
                for kk, vv in v.items()} if isinstance(v, Mapping) else v)
            for k, v in cfg_base.items()}
        )
        
        cfg_run.model.act_fn = act_fn
        cfg_run.optim.lr = lr
        cfg_run.optim.amort_lr = lr
        cfg_run.infer.mu_dt = mu_dt
        cfg_run.infer.num_train_iters = ntr
        cfg_run.infer.num_test_iters = nte
        cfg_run.infer.test_thresh = tth
        cfg_run.infer.train_thresh = tth
        cfg_run.exp.log_dir = f"results/sweeps/run_{i:05d}"
        main(cfg_run)
        i+=1


    hybrid = []
    paths = [f"run_{i:05d}" for i in range(500)]
    for p in paths:
        with open("results/sweeps/"+ p +"/0/metrics.json", "r", encoding="utf-8") as json_data:
            d = json.load(json_data)
        last_hybrid = d["hybrid_acc"][-1]
        #print(last_hybrid)
        if last_hybrid > 0.6:
            hybrid.append( (p, d["hybrid_acc"][-1]))

    print(hybrid)

'------------------METRICS---------------------------------------------'
def summarize_logits(logits, y_true, eps=1e-12, max_iter=200, lr=0.01):
    
    y_true = y_true.to(logits.device)
    # y to class indices
    if y_true.ndim == 1:
        y = y_true.long()
    else:
        y = torch.argmax(y_true, dim=1).long()
    # temperature aby sme to mohlu proovnvavat aj s hpc modelom
    logT = torch.zeros((), device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(set_to_none=True)
        T = torch.exp(logT) + 1e-12
        loss = F.cross_entropy(logits / T, y, reduction="mean")
        loss.backward()
        return loss
    opt.step(closure)
    T = float(torch.exp(logT).detach().cpu().item())
    #probability
    probs = F.softmax(logits / T, dim=1)
    pred = torch.argmax(probs, dim=1)
    p = probs.clamp_min(eps)
    
    conf = probs.max(dim=1).values 
    correct_mask = (pred == y)
    incorrect_mask = ~correct_mask
    return {
        "T": T,
        "acc": (pred == y).float().mean().item(),
        'loss':   F.cross_entropy(logits / T, y, reduction="mean").item(),
        "mean_prob_correct": conf[correct_mask].mean().item(),
        "mean_prob_incorrect": conf[incorrect_mask].mean().item(),
        "mean_entropy": -(p * p.log()).sum(dim=1).mean().item(),
    }

'-----------ROBUSTNESS------------------'
def add_gaussian_noise(x, sigma, mean=0.1307, std=0.3081):
    # denormalize 
    x01 = x * std + mean
    #add noise
    x01 = (x01 + sigma * torch.randn_like(x01)).clamp(0.0, 1.0)
    # back to normalized 
    return (x01 - mean) / std

def occlude_square( x, square, fill = 0.5 ):
    if x.ndim == 2:
        b, d = x.shape
        x_img = x.view(b, 1, 28, 28) #obrazok 28x28
        x_occ = occlude_square(x_img, square, fill=fill)
        return x_occ.view(b, d)
    
    b, c, h, w = x.shape
    s = min(square, h, w)
    g = torch.Generator(device=x.device)


    top = torch.randint(0, h - s + 1, (b,), generator=g, device=x.device)
    left = torch.randint(0, w - s + 1, (b,), generator=g, device=x.device)

    x2 = x.clone()
    for i in range(b):
        t = int(top[i].item())
        l = int(left[i].item())
        x2[i, :, t:t+s, l:l+s] = fill
    return x2

def load_hpc(hpc_ckpt_path, device):
    ckpt = torch.load(hpc_ckpt_path, map_location="cpu")
    extra = ckpt.get("extra", {})
    cfg_path = extra.get("config_path")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    #model podla json file teda hlavne architekrura
    model = HybridModel( nodes=cfg["model"]["nodes"],  amort_nodes=cfg["model"]["amort_nodes"],  mu_dt=cfg["infer"]["mu_dt"],
                         act_fn=utils.get_act_fn(cfg["model"]["act_fn"]), use_bias=cfg["model"]["use_bias"], kaiming_init=cfg["model"]["kaiming_init"] )

    #vahy
    model.load_checkpoint(hpc_ckpt_path, map_location="cpu")
    for l in model.layers: 
        l.weights = l.weights.to(device).float() 
        l.bias = l.bias.to(device).float() 
    
    for l in model.amort_layers: 
        l.weights = l.weights.to(device).float() 
        l.bias = l.bias.to(device).float()
    return model, cfg

def collect_logits( model, loader, device, standart = True,  x_transform = None, cfg = None, mode = None):
    '''
    :param standart: True ak ide o standart network, false ak ide o HPC
    :param mode: ma zmysel iba ak standart = False, mode = [amort, hybrid, pc]
    :param cfg: ma zmysel ak standart = False
    :x_transform: ma zmysel iba  ak standart = False
    '''
    with torch.no_grad():
        if not standart :
            
            infer = cfg["infer"]
            num_test_iters = int(infer["num_test_iters"])
            init_std = float(infer["init_std"])
            fixed_preds_test = bool(infer["fixed_preds_test"])
            test_thresh = infer["test_thresh"]
        logits_list, y_list = [], []
        
        for x, y in loader:
            x = x.to(device)
            if x_transform is not None:
                x = x_transform(x)
            
            if not standart:
                if x.ndim == 4:
                    x = x.view(x.size(0), -1)
                
                if mode == "amort":
                    logits = model.forward(x)
                elif mode == "hybrid" :
                    logits, _, _ = model.test_batch( x, num_iters=num_test_iters,init_std=init_std, fixed_preds=fixed_preds_test, use_amort=True, thresh=test_thresh )
                elif mode == "pc" :
                    logits, _, _ = model.test_batch( x, num_iters=num_test_iters, init_std=init_std, fixed_preds=fixed_preds_test, use_amort=False, thresh=test_thresh )
            else:
                logits = model(x)

            logits_list.append(logits.detach().cpu())
            y_list.append(y.detach().cpu())
    return torch.cat(logits_list, dim=0), torch.cat(y_list, dim=0)

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FC_DATAPATH = "results/standart_network"
    HPC_CKPT_PATH = "results/local_hybrid/0/checkpoints/epoch36.pth" 
    
    # MODELY
    fc_model = load_model_eval( dataset_name= 'MNIST', network_type='FC',  device=device,  datapath=FC_DATAPATH )
    hpc_model, hpc_cfg = load_hpc(HPC_CKPT_PATH, device=device)

    # DATA uz rovnako nacitane pouzivam pybrid loading
    fc_test_ds = datasets.SharedMNIST( train=False, normalize=True, flatten=False, return_one_hot=False, scale=None )
    hpc_test_ds = datasets.SharedMNIST(train=False, scale=hpc_cfg["data"]["label_scale"], size=hpc_cfg["data"]["test_size"], normalize=hpc_cfg["data"]["normalize"], flatten=True, return_one_hot=True)

    fc_test_loader = DataLoader(fc_test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=False)
    hpc_test_loader = datasets.get_dataloader(hpc_test_ds, batch_size=hpc_cfg["optim"]["test_batch_size"]) 

    report: Dict[str, dict] = {
        "paths": {"hpc": HPC_CKPT_PATH, "classic": FC_DATAPATH},
        "clean": {"fc": {}, "hpc": {"hybrid": {}, "pc": {}, "amort": {}}} ,
        "noise": {"fc": {}, "hpc": {"hybrid": {}, "pc": {}, "amort": {}}},
        "occlusion": {"fc": {}, "hpc": {"hybrid": {}, "pc": {}, "amort": {}}} }
   
    # ---------- CLEAN ----------
    fc_logits, fc_y = collect_logits(fc_model, fc_test_loader, device=device)
    report["clean"]["fc"] = summarize_logits(fc_logits, fc_y)
    for mode in ["hybrid", "pc", "amort"]:
        h_logits, h_y = collect_logits(hpc_model, hpc_test_loader, device=device, cfg=hpc_cfg, standart = False, mode= mode)
        report["clean"]["hpc"][mode] = summarize_logits(h_logits, h_y)
    
    # ---------- NOISE ----------
    for sigma in [0.0, 0.05, 0.1, 0.2]:
        tf = lambda x, s=sigma: add_gaussian_noise(x, s)

        fc_logits_n, fc_y_n = collect_logits(fc_model, fc_test_loader, device=device, x_transform=tf )
        report["noise"]["fc"][str(sigma)] = summarize_logits(fc_logits_n, fc_y_n)
        
        for mode in ["hybrid", "pc", "amort"]:
            h_logits_n, h_y_n = collect_logits( hpc_model, hpc_test_loader, device=device, cfg=hpc_cfg, x_transform=tf , standart = False, mode= mode)
            report["noise"]["hpc"][mode][str(sigma)] = summarize_logits(h_logits_n, h_y_n)
        
        # vyklresleniee
        x0, y0 = next(iter(hpc_test_loader))       
        x0 = x0.to(device)
        x0_occ = tf(x0) 
        datasets.plot_imgs(x0_occ, f"results/images/noise{sigma}.png")
   
    # # ---------- OCCLUSION ----------
    for sq in [8, 14, 20]:
        tf = lambda x, q=sq: occlude_square( x, square=q) #stvorcek randomne niekam

        fc_logits_o, fc_y_o = collect_logits(fc_model, fc_test_loader, device=device, x_transform=tf)
        report["occlusion"]["fc"][str(sq)] = summarize_logits(fc_logits_o, fc_y_o)
        for mode in ["hybrid", "pc", "amort"]:
            h_logits_o, h_y_o = collect_logits(hpc_model, hpc_test_loader, device=device, cfg=hpc_cfg, x_transform=tf, standart = False, mode= mode )
            report["occlusion"]["hpc"][mode][str(sq)] = summarize_logits(h_logits_o, h_y_o)
    
        # vyklresleniee
        x0, y0 = next(iter(hpc_test_loader))       
        x0 = x0.to(device)
        x0_occ = tf(x0) 
        datasets.plot_imgs(x0_occ, f"results/images/occlusion_sq{sq}.png")

    print(json.dumps(report, indent=2))
    with open('results/compare_report.json', "w") as f: json.dump(report, f, indent=2)


#HPC
cfg.exp.log_dir = "results/local_hybrid"
main(cfg)
#FC
run_training(datapath= 'results/standart_network')
run()

