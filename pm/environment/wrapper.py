from gymnasium import Wrapper, spaces
import random
import numpy as np

class EnvironmentWrapper(Wrapper):
    metadata = {"render_modes": []}  # Required for Gymnasium

    def __init__(self, env, transition_shape, seed=42):
        super().__init__(env)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.env = env
        self.num_stocks = len(env.stocks)

        action_shape = transition_shape["action"]["shape"][1:]
        action_type = transition_shape["action"]["type"]
        state_shape = transition_shape["state"]["shape"][1:]
        state_type = transition_shape["state"]["type"]
        print("action shape {}, action type {}, state shape {}, state type {}".format(action_shape, action_type, state_shape, state_type))

        self.action_space = spaces.Box(
            low=0,
            high=1.0,
            shape=action_shape,
            dtype=action_type,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=state_shape,
            dtype=state_type,
        )

    def reset(self, *, seed=None, options=None, **kwargs):
        if seed is not None:
            self.env.seed = seed
            random.seed(seed)
            np.random.seed(seed)
        obs, _ = self.env.reset(seed=seed, options=options)  # UNPACK inner info
        return obs, {}  # Proper Gymnasium format: (obs, info)

    def step(self, action):
        step_result = self.env.step(action)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            # fallback for older or non-gymnasium envs
            obs, reward, done, info = step_result
            terminated = done
            truncated = False

        return obs, reward, terminated, truncated, info
