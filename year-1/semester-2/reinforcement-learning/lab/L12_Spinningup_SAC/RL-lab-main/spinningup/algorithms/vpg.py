import torch
from spinup.utils.mpi_pytorch import mpi_avg_grads

# Set up function for computing VPG policy loss
def loss_pi_fn(data, ac):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = ac.pi(obs, act)
    loss_pi = -(logp * adv).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    pi_info = dict(kl=approx_kl, ent=ent)

    return loss_pi, pi_info
    
# Set up function for computing value loss
def loss_v_fn(data, ac):
    obs, ret = data['obs'], data['ret']
    return ((ac.v(obs) - ret)**2).mean()
    
def update_rule(data, pi_optimizer, vf_optimizer, logger, ac, train_v_iters):
    # Get loss and info values before update
    pi_l_old, pi_info_old = loss_pi_fn(data, ac)
    pi_l_old = pi_l_old.item()
    v_l_old = loss_v_fn(data, ac).item()

    # Train policy with a single step of gradient descent
    pi_optimizer.zero_grad()
    loss_pi, pi_info = loss_pi_fn(data, ac)
    loss_pi.backward()
    mpi_avg_grads(ac.pi)    # average grads across MPI processes
    pi_optimizer.step()

    # Value function learning
    for i in range(train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = loss_v_fn(data, ac)
        loss_v.backward()
        mpi_avg_grads(ac.v)    # average grads across MPI processes
        vf_optimizer.step()

    # Log changes from update
    kl, ent = pi_info['kl'], pi_info_old['ent']
    logger.store(LossPi=pi_l_old, LossV=v_l_old,
                 KL=kl, Entropy=ent,
                 DeltaLossPi=(loss_pi.item() - pi_l_old),
                 DeltaLossV=(loss_v.item() - v_l_old))
