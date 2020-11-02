# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import numpy as np
from utilities import soft_update, transpose_to_tensor, transpose_list, transpose_to_nested_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'



class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24*2+2+2=52
        self.maddpg_agent = [DDPGAgent(24, 256, 256, 2, 52, 256, 256),  
                             DDPGAgent(24, 256, 256, 2, 52, 256, 256)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=[0.0, 0.0]):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, float(noise_i)) for agent, obs, noise_i in zip(self.maddpg_agent, obs_all_agents, noise)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = []
        for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents):
            # a list of tensors is passed in the update loop, convert it to a single tensor of a list of inputs
            if len(obs)>1:
                obs = torch.from_numpy(np.vstack(obs)).float().to(device)
            target_actions.append(ddpg_agent.target_act(obs, noise))
        return target_actions

    def update(self, samples, agent_number, log):
        """update the critics and actors of all the agents """

        obs, obs_full, action, reward, next_obs, next_obs_full, done = zip(*samples)

        #input needs to be transposed from [samples of (agent 1 obs, agent 2 obs)] to [agent 1 (sample obs), agent 2 (sample obs)]
        agent1_next_obs, agent2_next_obs = zip(*next_obs)
        agent1_obs, agent2_obs = zip(*obs)
        obs = [torch.from_numpy(np.vstack(agent1_obs)).float().to(device), torch.from_numpy(np.vstack(agent2_obs)).float().to(device)]
        done = transpose_to_nested_list(done)
        reward = transpose_to_nested_list(reward)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act([agent1_next_obs, agent2_next_obs])
        
        #stack the various agents actions and observations to input to the critic
        target_actions = torch.cat(target_actions, 1)
        next_obs_full = torch.from_numpy(np.vstack(next_obs_full)).float().to(device)
        target_critic_input = torch.cat((next_obs_full,target_actions), 1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = torch.from_numpy(reward[agent_number]).float().to(device) + self.discount_factor * q_next * (1 - torch.from_numpy(done[agent_number].astype(np.uint8)).float().to(device))
        
        action = transpose_to_tensor(action)
        action = torch.cat(list(action), dim=1)
        critic_input = torch.cat((torch.from_numpy(np.vstack(obs_full)).float().to(device), action),1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        obs_full = torch.from_numpy(np.vstack(obs_full)).float().to(device)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        log.info('agent{}: critic loss\t{}\tactor loss\t{}'.format(agent_number,cl,al))

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
                  
    def adapt_learning_rate(self, ratio = 0.999):
        for agent in self.maddpg_agent:
            for group in agent.critic_optimizer.param_groups:
                group['lr'] = max( 1e-7, group['lr']*ratio)
            for group in agent.actor_optimizer.param_groups:
                group['lr'] = max( 1e-7, group['lr']*ratio)

    def reset_noise(self):
        for agent in self.maddpg_agent:
            agent.noise.reset()



