import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams["figure.figsize"] = (20,8)

# def plot_rewards_loss(rewards, losses, value_losses, num_trajectories=300, log_scale=True):

#     print(rewards.shape, losses.shape, value_losses.shape)

#     datapoints = len(rewards)

#     x = np.arange(0, num_trajectories * datapoints, num_trajectories)

#     # Visualize the data
#     fig, ax1 = plt.subplots()

#     ax1.plot(x, rewards, 'orange', linewidth=2,label='Rewards')
#     ax1.set_xlabel('Trajectories')
#     ax1.set_ylabel('Reward')
    
#     if log_scale:
#         ax1.set_yscale('log')

#     ax2 = ax1.twinx()
#     ax2.plot(x, losses, alpha=0.3, label='Loss')
#     ax2.plot(x, value_losses, alpha=0.3, color="green", label='Value Loss')
#     ax2.set_ylabel('Loss')
    
#     if log_scale:
#         ax2.set_yscale('log')

#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

#     plt.title('Loss and Rewards')

#     plt.show()


def plot_rewards_loss(rewards, losses, value_losses, num_trajectories=300, log_scale=False):

    print(rewards.shape, losses.shape, value_losses.shape)

    datapoints = len(rewards)

    x = np.arange(0, num_trajectories * datapoints, num_trajectories)

    # Visualize the data
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    ax1.plot(x, rewards, 'orange', linewidth=2,label='Rewards')
    ax1.set_xlabel('Trajectories')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards')
    if log_scale:
        ax1.set_yscale('log')

    ax2.plot(x, losses, alpha=0.3, label='Loss')
    ax2.set_xlabel('Trajectories')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss')
    if log_scale:
        ax2.set_yscale('log')

    ax3.plot(x, value_losses, alpha=0.3, color="green", label='Value Loss')
    ax3.set_xlabel('Trajectories')
    ax3.set_ylabel('Value Loss')
    ax3.set_title('Value Loss')
    if log_scale:
        ax3.set_yscale('log')

    plt.tight_layout()
    plt.show()



# def plot_states(states,actions,optimum_storage_capacity, id=0):

#     time = np.arange(len(states))
#     optimum_storage = np.zeros(len(states))
#     optimum_storage[:] = optimum_storage_capacity

#     fig, ax1 = plt.subplots()

#     ax1.plot(time,states, 'orange', linewidth=2,label='Thermal Energy Storage Capacity')
#     ax1.plot(time, optimum_storage, 'g', linestyle='--', label='Optimum Storage Capacity')

#     ax1.set_xticks(time)
#     new_labels = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#     label_mapping = {i: label for i, label in enumerate(new_labels)}
#     x_labels = ['' if i % 4 != 0 else label_mapping[i//4] for i in time]
#     ax1.set_xticklabels(x_labels)
    


#     ax1.set_xlabel('Time (hours)')
#     ax1.set_ylabel('Storage Capacity (kwH)')
#     ax1.set_ylim(0, 19)

#     ax2 = ax1.twinx()
#     ax2.bar(time, actions, alpha=0.3, label='Action (Heat Up)')
#     ax2.set_ylabel('Action (0 or 1)')
#     ax2.set_ylim(0, 1)

#     # Combine legends
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
#     plt.title('Thermal Energy Storage Capacity and Action')


#     plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/img/' + str(id) +'.png')
#     plt.show()

def plot_states(states, actions, optimum_storage_capacity, id=0):
    time = np.arange(len(states))
    optimum_storage = np.zeros(len(states))
    optimum_storage[:] = optimum_storage_capacity

    fig, ax1 = plt.subplots()

    ax1.plot(time, states, 'orange', linewidth=2, label='Thermal Energy Storage Capacity')
    ax1.plot(time, optimum_storage, 'g', linestyle='--', label='Optimum Storage Capacity')
    
    ax1.set_xticks(time)
    new_labels = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    label_mapping = {i: label for i, label in enumerate(new_labels)}
    
    if len(states) == 96:
        # For 15-minute intervals
        x_labels = ['' if i % 4 != 0 else label_mapping[i//4] for i in time]
    elif len(states) <= 24:
        # For 1-hour intervals
        x_labels = [label_mapping[i] for i in time]
    else:
        raise ValueError("Unsupported time interval. Expected 24 or 96.")
    
    ax1.set_xticklabels(x_labels)

    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Storage Capacity (kWh)')
    ax1.set_ylim(0, 19)

    ax2 = ax1.twinx()
    ax2.bar(time, actions, alpha=0.3, label='Action (Heat Up)')
    ax2.set_ylabel('Action (0 or 1)')
    ax2.set_ylim(0, 1)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title('Thermal Energy Storage Capacity and Action')

    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/img/' + str(id) + '.png')
    plt.show()
