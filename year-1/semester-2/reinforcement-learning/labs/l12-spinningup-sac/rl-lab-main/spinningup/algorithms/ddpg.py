import torch

# Set up function for computing DDPG Q-loss
def loss_q_fn(data, ac, ac_targ, gamma):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    q = ac.q(o,a)

    # Bellman backup for Q function
    with torch.no_grad():
        q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            
        #TODO compute Bellman backup
        backup = #TODO

    #TODO compute MSE loss against Bellman backup
    loss_q = #TODO

    # Useful info for logging
    loss_info = dict(QVals=q.detach().numpy())

    return loss_q, loss_info

# Set up function for computing DDPG pi loss
def loss_pi_fn(data, ac):
    o = data['obs']
    #TODO compute the q-value for the action provided by the policy on observation o
    q_pi = #TODO
    return -q_pi.mean()
    
def update_rule(data, q_optimizer, pi_optimizer, logger, ac, ac_targ, gamma, polyak):
    # First run one gradient descent step for Q.
    q_optimizer.zero_grad()
    loss_q, loss_info = loss_q_fn(data, ac, ac_targ, gamma)
    loss_q.backward()
    q_optimizer.step()

    # Freeze Q-network so you don't waste computational effort 
    # computing gradients for it during the policy learning step.
    for p in ac.q.parameters():
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi = loss_pi_fn(data, ac)
    loss_pi.backward()
    pi_optimizer.step()

    # Unfreeze Q-network so you can optimize it at next DDPG step.
    for p in ac.q.parameters():
        p.requires_grad = True

    # Record things
    logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)
