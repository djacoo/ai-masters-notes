import torch
from spinup.utils.mpi_tools import mpi_avg
from spinup.utils.mpi_pytorch import mpi_avg_grads

# Set up function for computing PPO policy loss
def loss_pi_fn(data, ac, clip_ratio):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy probability for state obs and action act
    pi, logp = ac.pi(obs, act)
        
    #TODO compute the policy function loss
    #ratio = #TODO
    #clip_adv = #TODO
    #loss_pi = #TODO

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

# Set up function for computing value loss
def loss_v_fn(data, ac):
    obs, ret = data['obs'], data['ret']
        
    #TODO compute the value function loss
    #loss_v = #TODO
        
    return loss_v
    
def update_rule(data, pi_optimizer, vf_optimizer, logger, ac, clip_ratio, train_pi_iters, train_v_iters, target_kl):
    pi_l_old, pi_info_old = loss_pi_fn(data, ac, clip_ratio)
    pi_l_old = pi_l_old.item()
    v_l_old = loss_v_fn(data, ac).item()

    # Train policy with multiple steps of gradient descent
    for i in range(train_pi_iters):
        pi_optimizer.zero_grad()
        loss_pi, pi_info = loss_pi_fn(data, ac, clip_ratio)
        kl = mpi_avg(pi_info['kl'])
        if kl > 1.5 * target_kl:
            logger.log('Early stopping at step %d due to reaching max kl.'%i)
            break
        loss_pi.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

    logger.store(StopIter=i)

    # Value function learning
    for i in range(train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = loss_v_fn(data, ac)
        loss_v.backward()
        mpi_avg_grads(ac.v)    # average grads across MPI processes
        vf_optimizer.step()

    # Log changes from update
    kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    logger.store(LossPi=pi_l_old, LossV=v_l_old,
                 KL=kl, Entropy=ent, ClipFrac=cf,
                 DeltaLossPi=(loss_pi.item() - pi_l_old),
                 DeltaLossV=(loss_v.item() - v_l_old))
