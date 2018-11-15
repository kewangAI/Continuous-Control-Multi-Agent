from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent


def ddpg(env, agent,n_episodes=1000, max_t=300, print_every=1):
    brain_name = env.brain_names[0]
    #brain = env.brains[brain_name]
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        agent_num = env_info.vector_observations.shape[0]
        score = np.zeros((agent_num,))
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += np.array(rewards)
            if any(dones):
                break

        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

        if i_episode % print_every == 0:
            print('\rEpisode {} \tCurrent Score: {:.2f} \tAverage Score over 100 Eps: {:.2f}'.format(i_episode, scores[-1], np.mean(scores_deque)))
            if (i_episode > 100) and (np.mean(scores_deque) >= 30) :
                print('Problem Solved!')
                break

    return scores

def test_agent(agent, brain_name):
    total_reward =0
    env_info = env.reset()[brain_name]
    state = env_info.vector_observations[0]
    for j in range(200):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        total_reward += reward
        done = env_info.local_done[0]
        if done:
            break
    return total_reward

def rolling_aver(scores):
    scores_deque = deque(maxlen=100)
    scores_100 = []
    for s in scores:
        scores_deque.append(s)
        scores_100.append(np.mean(scores_deque))

    return scores_100

def plot_save_score(scores, file_name):
    scores_100 = rolling_aver(scores)
    v_scores = np.array([range(1, len(scores)+1), scores, scores_100])
    np.savetxt(file_name, np.transpose(v_scores), delimiter=',')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig.savefig("training.pdf", bbox_inches='tight')


if __name__ == '__main__':

    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=3)

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)


    scores = ddpg(env, agent)

    plot_save_score(scores, 'scores.csv')


    env.close()