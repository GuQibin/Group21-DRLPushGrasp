"""
Test script for Strategic Push-Grasp Environment.

This is a simple test harness that creates the environment and runs random
actions to verify basic functionality. Used for:
- Smoke testing after environment changes
- Verifying rendering works
- Checking episode reset logic
- Validating action/observation spaces
"""


import gymnasium as gym
import envs  # This line is CRITICAL - executes registration code in __init__.py

def main():
    """
    Main test function that runs the environment with random actions.
    
    Test Procedure:
    1. Create environment with GUI rendering enabled
    2. Reset to get initial observation
    3. Run 1000 random actions (multiple episodes)
    4. Reset whenever episode terminates or truncates
    5. Close environment and cleanup
    
    Purpose:
    This is NOT for training - it's a functional test to verify:
    - Environment can be instantiated via gym.make()
    - Rendering works without crashes
    - Action space sampling works
    - Step function returns correct tuple format
    - Reset works after episode completion
    - No crashes occur during extended operation
    
    Expected Behavior:
    - Window opens showing PyBullet GUI
    - Robot performs random movements (no learning)
    - Objects spawn and get manipulated randomly
    - Episodes reset when terminated or truncated
    - Script completes without errors
    """
    env = gym.make("StrategicPushAndGrasp-v0", render_mode="human")

    # Reset the environment to get initial state
    # Returns:
    # - obs: initial observation (state representation)
    # - info: additional diagnostic information
    obs, info = env.reset()
    
    print("Starting test with random actions...")
    # Execute 1000 random actions to test environment stability
    for _ in range(1000):
        # Sample a random action from the environment's action space
        # This tests that action space is properly defined and sample-able
        action = env.action_space.sample()

        # Execute the action in the environment
        # Returns a tuple of 5 values (Gymnasium API):
        # - obs: new observation after taking the action
        # - reward: reward signal from the environment
        # - terminated: whether episode ended due to success/failure condition
        # - truncated: whether episode ended due to time limit or other constraint
        # - info: additional environment information
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if episode has ended (either terminated or truncated)
        # In older Gym versions, this was a single 'done' flag
        done = terminated or truncated

        # If episode ended, reset the environment to start a new episode
        # This tests the reset functionality and environment reinitialization
        if done:
            obs, info = env.reset()

    # Properly close the environment and cleanup resources
    env.close()
    print("Test finished.")



if __name__ == "__main__":
    # Entry point - run the main function when script is executed directly
    main()
