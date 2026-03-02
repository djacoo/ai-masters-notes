import torch

# Set up function for computing SAC Q-losses
def loss_q_fn(data, ac, ac_targ, gamma, alpha):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    q1 = ac.q1(o,a)
    q2 = ac.q2(o,a)

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        a2, logp_a2 = ac.pi(o2)

        # TODO compute Target Q-values
        q1_pi_targ = # TODO
        q2_pi_targ = # TODO
            
        #TODO compute Bellman backup
        q_pi_targ = #TODO
        backup = #TODO

    #TODO compute MSE loss against Bellman backup
    loss_q1 = #TODO
    loss_q2 = #TODO
    loss_q = #TODO

    # Useful info for logging
    q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

    return loss_q, q_info

# Set up function for computing SAC pi loss
def loss_pi_fn(data, ac, alpha):
    o = data['obs']
    pi, logp_pi = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    #TODO compute entropy-regularized policy loss
    loss_pi = # TODO

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().numpy())

    return loss_pi, pi_info
    
def update_rule(data, q_optimizer, pi_optimizer, q_params, logger, ac, ac_targ, gamma, alpha, polyak):
    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, q_info = loss_q_fn(data, ac, ac_targ, gamma, alpha)
    loss_q.backward()
    q_optimizer.step()

    # Record things
    logger.store(LossQ=loss_q.item(), **q_info)

    # Freeze Q-networks so you don't waste computational effort 
    # computing gradients for them during the policy learning step.
    for p in q_params:
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi, pi_info = loss_pi_fn(data, ac, alpha)
    loss_pi.backward()
    pi_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
        p.requires_grad = True

    # Record things
    logger.store(LossPi=loss_pi.item(), **pi_info)

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)
