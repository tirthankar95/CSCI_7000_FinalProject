from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class WindyEnv(MiniGridEnv):
    """
    Environment with a wind
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())


        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))
        #self.place_agent(top=(1,1))
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a yellow key on the left side
        self.place_obj(
            obj=Wind(1),

        )
        self.place_obj(
            obj=Wind(3),

        )

        self.mission = "Never touch the wind, it throws into a random direction"

class WindyEnv5x5(WindyEnv):
    def __init__(self):
        super().__init__(size=5)

class WindyEnvEnv6x6(WindyEnv):
    def __init__(self):
        super().__init__(size=6)

class WindyEnvEnv16x16(WindyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-Windy-5x5-v0',
    entry_point='gym_minigrid.envs:WindyEnvEnv5x5'
)

register(
    id='MiniGrid-WindyEnv-6x6-v0',
    entry_point='gym_minigrid.envs:WindyEnv6x6'
)

register(
    id='MiniGrid-Windy-8x8-v0',
    entry_point='gym_minigrid.envs:WindyEnv'
)

register(
    id='MiniGrid-Windy-16x16-v0',
    entry_point='gym_minigrid.envs:WindyEnv16x16'
)
