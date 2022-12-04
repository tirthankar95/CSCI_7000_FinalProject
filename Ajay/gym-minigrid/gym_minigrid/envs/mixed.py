from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import itertools as itt


class MixedEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward, .
    """

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, seed=None):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        super().__init__(
            grid_size=size,
            max_steps=10*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )
        #print(self.obstacle_type,obstacle_type)

    def _gen_grid(self, width, height):
        #assert width % 2 == 1 and height % 2 == 1  # odd size
        #print(self.obstacle_type)
        # Create an empty grid
        self.grid = Grid(width, height)
        #print(width/2 - 2,height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Create a vertical splitting wall
        #splitIdx = self._rand_int(2, width/2 - 2)
        splitIdx = 2

        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        #self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        # doorIdx = self._rand_int(1, height - 2)
        doorIdx = 4
        self.grid.set(splitIdx, doorIdx, Door('yellow', is_locked=True))

        # Place a yellow key on the left side
        # self.place_obj(
        #     obj=Key('yellow'),
        #     #obj = Wind(),
        #     top=(0, 0),
        #     size=(splitIdx, height)
        # )
        pos = np.array((1, height - 3))
        self.grid.set(*pos, Key('yellow'))

        # self.place_obj(
        #     obj = Wind(1)
        # )
        # self.place_obj(
        #     obj=Wind(2)
        # )

        #pos = np.array((splitIdx+3, doorIdx))
        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(splitIdx+2, height - 2, 2)]
        rivers += [(h, j) for j in range(doorIdx+2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        rivers_h = [splitIdx+3]
        rivers_v = [doorIdx+2]
        obstacle_pos = itt.chain(
            itt.product(range(doorIdx+1, width - 1), rivers_h),
            itt.product(rivers_v, range(splitIdx+1, height - 1)),
        )
        #print(obstacle_pos, rivers_h,rivers_v)
        #print(self.obstacle_type)
        for i, j in obstacle_pos:
            #self.grid.set(i, j, self.obstacle_type())
            self.grid.set(i, j, self.obstacle_type())

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)
        #print("Split, Door",splitIdx,doorIdx)
        # Create openings
        limits_v = [splitIdx+1] + rivers_v + [height - 1]
        limits_h = [doorIdx] + rivers_h + [width - 1]
        #print("Limits",limits_v, limits_h,len(rivers_v), len(rivers_h))
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )


class SimpleMixedEnv(MixedEnv):
    def __init__(self):
        super().__init__(size=15, num_crossings=1, obstacle_type=Lava)

class MixedEnvS15N2Env(MixedEnv):
    def __init__(self):
        super().__init__(size=15, num_crossings=2, obstacle_type=Lava)

class MixedEnvS15N3Env(MixedEnv):
    def __init__(self):
        super().__init__(size=15, num_crossings=3, obstacle_type=Lava)

class MixedEnvS21N5Env(MixedEnv):
    def __init__(self):
        super().__init__(size=21, num_crossings=5, obstacle_type=Lava)

register(
    id='MiniGrid-SimpleMixedEnvS9N1-v0',
    entry_point='gym_minigrid.envs:SimpleMixedEnv'
)

register(
    id='MiniGrid-MixedEnvS9N2-v0',
    entry_point='gym_minigrid.envs:MixedEnvS9N2Env'
)

register(
    id='MiniGrid-MixedEnvS9N3-v0',
    entry_point='gym_minigrid.envs:MixedEnvS9N3Env'
)

register(
    id='MiniGrid-MixedEnvS11N5-v0',
    entry_point='gym_minigrid.envs:MixedEnvS11N5Env'
)
