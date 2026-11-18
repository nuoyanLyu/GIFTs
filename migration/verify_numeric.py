import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch, numpy as np, ast, types, runpy, json, argparse, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--legacy-root',type=Path,default=root.parent/'RAGEN/verl')
args=parser.parse_args()
new=runpy.run_path(str(root/"ragen/trainer/core_algos.py"))
helper={"torch":torch}
s=(args.legacy_root/'verl/utils/torch_functional.py').read_text()
names=["masked_mean","masked_var","masked_whiten","clip_by_value","entropy_from_logits"]
for n in ast.parse(s).body:
 if isinstance(n,ast.FunctionDef) and n.name in names:
  exec(compile(ast.Module(body=[n],type_ignores=[]),"legacy_helper","exec"),helper)
old={}
exec((args.legacy_root/'verl/trainer/ppo/core_algos.py').read_text(),old)
old["verl_F"]=types.SimpleNamespace(**{n:helper[n] for n in names})
torch.manual_seed(17)
checks=[]
def check(name,kwargs):
 def copy_args():return {k:v.clone().requires_grad_(v.requires_grad) if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
 a,b=old[name](**copy_args()),new[name](**copy_args())
 if not isinstance(a,tuple):a,b=(a,),(b,)
 for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0,equal_nan=True)
 checks.append(name)
mask=torch.tensor([[1,1,0,1],[1,0,1,1],[1,1,1,0],[1,1,1,1]],dtype=torch.float64)
r=torch.randn(4,4,dtype=torch.float64)*mask
v=torch.randn(4,4,dtype=torch.float64)
check("compute_gae_advantage_return",dict(token_level_rewards=r,values=v,response_mask=mask,gamma=.93,lam=.87))
for idx in [np.array(["a","b","c","d"]),np.array(["a","a","b","b"])]:
 for norm in [True,False]:check("compute_grpo_outcome_advantage",dict(token_level_rewards=r,response_mask=mask,index=idx,norm_adv_by_std_in_grpo=norm))
for mode in ["token-mean","seq-mean-token-sum","seq-mean-token-mean","seq-mean-token-sum-norm"]:
 oldp=torch.randn(4,4,dtype=torch.float64)
 logp=torch.randn(4,4,dtype=torch.float64)
 # Include a log-ratio outside the new upstream clamp, so a default-loss switch is detectable.
 logp[0,0]=oldp[0,0]+25
 adv=torch.randn(4,4,dtype=torch.float64);adv[0,0]=-1
 kw=dict(old_log_prob=oldp,log_prob=logp,advantages=adv,response_mask=mask,cliprange=.2,loss_agg_mode=mode)
 check("compute_policy_loss",kw)
 grad=[]
 for ns in [old,new]:
  x=logp.clone().requires_grad_();loss=ns["compute_policy_loss"](**{**kw,"log_prob":x})[0];loss.backward();grad.append(x.grad)
 torch.testing.assert_close(*grad,rtol=0,atol=0,equal_nan=True);checks.append("policy_gradient_"+mode)
check("compute_value_loss",dict(vpreds=v,returns=r,values=v+.3,response_mask=mask,cliprange_value=.2))
for typ in ["kl","abs","mse","low_var_kl"]:check("kl_penalty",dict(logprob=v,ref_logprob=r,kl_penalty=typ))
assert not torch.cuda.is_initialized()
print(json.dumps({"checks":len(checks),"python":sys.executable,"prefix":sys.prefix,"bitwise_equal":True,"cuda_initialized":torch.cuda.is_initialized(),"singleton_adv_nonzero": bool(new["compute_grpo_outcome_advantage"](r,mask,np.array(["a","b","c","d"]))[0].abs().sum()>0)}))
