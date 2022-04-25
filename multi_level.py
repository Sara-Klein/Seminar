import numpy as np
from scipy.special import softmax
import math
from numpy.random import default_rng
import matplotlib 
import matplotlib.pyplot as plt
from four_connect import Connect4Env
from network_policy import PolicyNet
import torch
# Set global variables for Multilevel Estimation
TIME = 5
M = 2 
R = 2
L = 5
rng = default_rng()

class Multilevel():
    def __init__(self, env=Connect4Env()) -> None:
        self.env = env
        self.theta_0 = torch.ones(len(self.env.observation_space))
        self.m = M
        self.sigma_0 = 1
        self.eps_0 = 1
        self.rho = (1/2) * (1+R)
        self.gamma_0 = (1+R)/L

    # Not needed in more complex model
    def grad_soft(self, theta):

        return np.eye(len(theta),len(theta))- np.matmul(softmax(theta).reshape(len(theta),1), np.ones((1,len(theta))))

    def sample_env(self, theta, t, state):
        # TODO: Sollten wir in der Theorie nicht samplen bis das Spiel immer vorbei ist? Es solllte jetzt schneller gehen
        # Ich kann auch versuchen Offline Reinforcement Learning zu etablieren
        rewards = 0
        if action == None:
            trajectory = [state]
            for n in range((TIME-t)):
                action = int(np.random.choice(self.env.action_space.n,1,p=theta))
                trajectory.append(action)
                obs, reward, done, _ = self.env.step(action)
                trajectory.append(obs)
                rewards += reward
        else:
            trajectory = [state, action]
            for n in range((TIME-t)):
                trajectory.append(self.env.step(action))
                action = int(np.random.choice(self.env.action_space.n,1,p=theta))
                trajectory.append(action)
                obs, reward, done, _ = self.env.step(action)
                trajectory.append(obs)
                rewards += reward
        return trajectory, rewards

    # def sample(self, theta, t, state, action):

    #     if action == None:
    #         trajectory = [state]
    #         for n in range((TIME-t)):
    #             action = np.random.choice(actions, p=softmax(theta), size=1)[0]
    #             trajectory.append(action)
    #             state = rng.binomial(1, p=(1-transition[state, action]), size=1)[0]
    #             trajectory.append(state)
    #     else:
    #         trajectory = [state, action]
    #         for n in range((TIME-t)):
    #             state = rng.binomial(1, p=(1-transition[state, action]), size=1)[0]
    #             trajectory.append(state)
    #             action = np.random.choice(actions, p=softmax(theta), size=1)[0]
    #             trajectory.append(action)

    #     return trajectory


    #Das ist unser Code
    def multi_level_step(self):
        # self.theta = self.theta_0.copy()
        self.theta = self.theta_0.clone().detach()
        print(self.theta)
        for n in range(1,5): # 5 frei gewählt, um schnelle Resultate zu sehen

            eps_n = self.eps_0 * 1/(n**self.rho)
            sigma_n = 1/(n**R) * self.sigma_0
            gamma_n = self.gamma_0 * 1/n
            m_n = 2*max(math.ceil((np.log(1/eps_n)) / np.log(M)), 1)
            #print(m_n)
            L_m = np.zeros(len(self.theta))
            total_acts = list()
            for k in range(1,m_n+1):
                #N_nk = 100* math.ceil((1/np.sqrt(sigma_n))**2 * m_n * M**(-k))
                if n ==1: 
                    N_nk = math.ceil(1000* (1/np.sqrt(sigma_n))**2 * m_n * M**(-k))
                else: 
                    N_nk = math.ceil(10*(1/np.sqrt(sigma_n))**2 * m_n * M**(-k))
                #print(N_nk)
                L_k = np.zeros(len(self.theta))
                for l in range(1,N_nk+1):
                    traject, _ = self.sample_env(self.theta,t=0, state=0, action=None)
                    acts = list()
                    Q_k = list()
                    Q_k2 = list()
                    for t in range(TIME-1):
                        state = traject[2*t]
                        action = traject[2*t+1]
                        acts.append(action)
                        rew_to_go = 0
                        for u in range(M**k):
                            _, rew_to_go = self.sample_env(self.theta, state=state, action=action)
                            # t_to_go = self.sample_env(self.theta, t, state=state, action=action)
                            # rew = np.array(t_to_go).reshape(int((len(t_to_go))/2),2)
                            # for tup in rew[0:]:
                            #     rew_to_go += rewards[tup[0],tup[1]]
                        if k ==1:
                            Q_k.append(1/(M**k) * rew_to_go)   
                            Q_k2.append(0)
                        else: 
                            rew_to_go2 = 0
                            for u in range(M**(k-1)):
                                _,rew_to_go2 = self.sample_env(self.theta, t, state=state, action=action)
                                # rew = np.array(t_to_go).reshape(int((len(t_to_go))/2),2)
                                # for tup in rew[0:]:
                                #     # think about how to model the reward with the environment
                                #     rew_to_go2 += rewards[tup[0],tup[1]]
                            Q_k.append(1/(M**k) * rew_to_go)
                            Q_k2.append(1/(M**(k-1)) * rew_to_go2)
                    total_acts.append(acts)  

                    # TODO: Network should be depended of state, need to check if it works correctly, idea: do forward pass to get actions, 
                    L_k = L_k +  np.sum(np.multiply(PolicyNet(self.env.observation_space.shape, self.env.action_space.n).backward(gradient=self.theta)[acts],(np.array(Q_k)-np.array(Q_k2))[:,np.newaxis]), axis=0)
                    # L_k = L_k +  np.sum(np.multiply(self.grad_soft(self.theta)[acts],(np.array(Q_k)-np.array(Q_k2))[:,np.newaxis]), axis=0)
                L_m = L_m +((1/N_nk) * L_k)
            plt.hist(np.array(total_acts).reshape(np.product(np.shape(total_acts))))
            plt.show()
            print(np.sum(L_m))
            self.theta = self.theta +  torch.from_numpy(gamma_n * L_m)
        return self.theta


if __name__ == '__main__':
    game = Multilevel()
    params = game.multi_level_step()
    print(f'The resutling params are: {params}')