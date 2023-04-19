import matplotlib.pyplot as plt

def plot_rewards_loss(rewards, losses):



    # Visualize the data
    fig, ax1 = plt.subplots()

    ax1.plot(rewards, 'orange', linewidth=2,label='Rewards')
    ax1.set_ylabel('Reward')

    ax2 = ax1.twinx()
    ax2.plot(losses, alpha=0.3, label='Loss')
    ax2.set_ylabel('Loss')

    plt.show()