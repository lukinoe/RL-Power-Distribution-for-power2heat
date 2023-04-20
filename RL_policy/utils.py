import matplotlib.pyplot as plt
import numpy as np

def plot_rewards_loss(rewards, losses):



    # Visualize the data
    fig, ax1 = plt.subplots()

    ax1.plot(rewards, 'orange', linewidth=2,label='Rewards')
    ax1.set_ylabel('Reward')

    ax2 = ax1.twinx()
    ax2.plot(losses, alpha=0.3, label='Loss')
    ax2.set_ylabel('Loss')

    plt.show()


def plot_states(states,actions,optimum_storage_capacity):

    time = np.arange(len(states))
    optimum_storage = np.zeros(len(states))
    optimum_storage[:] = optimum_storage_capacity


    fig, ax1 = plt.subplots()

    ax1.plot(time,states, 'orange', linewidth=2,label='Thermal Energy Storage Capacity')
    ax1.plot(time, optimum_storage, 'g', linestyle='--', label='Optimum Storage Capacity')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Storage Capacity (kWh)')
    ax1.set_ylim(0, 19)

    ax2 = ax1.twinx()
    ax2.bar(time, actions, alpha=0.3, label='Action (Heat Up)')
    ax2.set_ylabel('Action (0 or 1)')
    ax2.set_ylim(0, 1)

    # # Combine legends
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.title('Thermal Energy Storage Capacity and Action')
    plt.show()