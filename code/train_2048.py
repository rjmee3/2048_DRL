from agent import Agent
from state import State

# Define constants
INPUT_CHANNELS = 1
OUTPUT_SIZE = 4  # Assuming there are 4 possible actions (left, right, up, down)
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 10

# Create an instance of the Agent class
agent = Agent(INPUT_CHANNELS, OUTPUT_SIZE, LEARNING_RATE, GAMMA)

# Create an instance of the State class
state = State()

# Training loop
epsilon = EPSILON_START
for episode in range(1000):
    state.reset()  # Reset the environment for a new episode
    total_reward = 0

    while not state.game_over:
        # Select an action using epsilon-greedy policy
        action = agent.select_action(state.one_hot_encode(), epsilon)

        # Take the selected action and observe the next state and reward
        next_state = state.board.copy()  # Replace this with your environment's state representation
        state.move(action)  # Perform the action in the environment
        reward = state.score - state.prev_score  # Calculate the reward

        # Store the experience in the agent's memory
        agent.train(state.one_hot_encode(), action, next_state, reward, state.game_over)

        total_reward += reward

    # Update the target DQN periodically
    if episode % TARGET_UPDATE_FREQUENCY == 0:
        agent.update_target_dqn()

    # Decay epsilon for exploration-exploitation trade-off
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Print the total reward for the episode
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

for _ in range(10):
    state.reset()
    while not state.game_over:
        action = agent.select_action(state.one_hot_encode(), epsilon=0)
        state.move(action)
        state.print()
        print("-----")
