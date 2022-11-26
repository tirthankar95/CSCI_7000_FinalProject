import gym
from gym import spaces
import pygame
import numpy as np
import random


class GridWorldEnvTm(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=6,g=1,t=-1,w=0,p=1):
        self.size = size  # The size of the square grid
        self.window_size = 256  # The size of the PyGame window
        self.g=g
        self.t=t
        self.w=w 
        self.p=p
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target_goal": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target_trap": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "block": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "diverted": spaces.Box(0,1, shape=(1,), dtype=bool),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self.windyAction = [{0:0.7,1:0.15,3:0.15},{1:0.7,0:0.15,2:0.15},\
                            {2:0.7,1:0.15,3:0.15},{3:0.7,0:0.15,2:0.15}]
        self.diverted=False
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location,"target_goal":self._target_location0,\
                "target_trap":self._target_location1,"block":self._block,"diverted":np.array(self.diverted).reshape(1,)}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location0, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array((1,3)) # (row,col) are reversed. 

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location0 = np.array((3,0))
        self._target_location1 = np.array((3,1))
        self._block=np.array((1,1))
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        self.diverted=False
        return observation, info

    def step(self, action):
        allActions=self.windyAction[action]
        actionN=random.choice([x for x in allActions for y in range((int)(100*allActions[x]))])
        self.diverted=False 
        if actionN!=action:
            self.diverted=True 
        action=actionN 
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        _agent_locationNew = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if np.array_equal(_agent_locationNew,self._block):
            _agent_locationNew=self._agent_location
        self._agent_location=_agent_locationNew
        # An episode is done iff the agent has reached the target
        terminated0 = np.array_equal(self._agent_location, self._target_location0)
        terminated1 = np.array_equal(self._agent_location, self._target_location1)
        reward=self.w 
        if terminated0:
            reward=self.g 
        if terminated1:
            reward=self.t 
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, (terminated0 or terminated1), False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target (good)
        pygame.draw.rect(
            canvas,
            (14, 255, 87),
            pygame.Rect(
                pix_square_size * self._target_location0,
                (pix_square_size, pix_square_size),
            ),
        )
        # Second we draw the target (bad)
        pygame.draw.rect(
            canvas,
            (255, 106, 6),
            pygame.Rect(
                pix_square_size * self._target_location1,
                (pix_square_size, pix_square_size),
            ),
        )
        # Third we draw the block
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                pix_square_size * self._block,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
