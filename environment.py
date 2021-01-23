import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, field_size):
        self.history = []
        self.position = position
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"

def get_target_foods_from_str(target_foods_str):

    target_foods = []
    for element in range(0, len(target_foods_str)):
        target_foods.append(int(target_foods_str[element]))

    return target_foods

class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        field_size,
        n_food,
        n_food_cat,
        target_food,
        sight,
        max_episode_steps,
        force_coop
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.n_food = n_food
        self.n_food_cat = n_food_cat
        self.food_types = list(range(1, n_food_cat + 1))
        self.target_foods = get_target_foods_from_str(target_food)
        self.n_target_foods = len(self.target_foods)*(self.n_food/self.n_food_cat)

        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self.viewer = None

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        field_x = self.field.shape[1]
        field_y = self.field.shape[0]

        min_obs = [-1, -1, 0] * self.n_food + [0, 0] * len(self.players)
        max_obs = [field_x, field_y, self.n_food_cat] * self.n_food + [field_x, field_y] * len(self.players)

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):

        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1
        else:
            return None

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self):

        if self.n_food % self.n_food_cat != 0:
            raise ValueError("Unknown food configuration")

        foods_per_type = int(self.n_food / self.n_food_cat)
        food_count_total = 0

        for food_type in range(self.n_food_cat):

            food_count_type = 0
            attempts = 0

            while food_count_type < foods_per_type and attempts < 100:
                attempts += 1
                row = self.np_random.randint(1, self.rows - 1)
                col = self.np_random.randint(1, self.cols - 1)

                # check if it has neighbors:
                if not self._is_empty_location(row, col):
                    continue
                elif len(self.adjacent_players(row, col)) == self.n_agents:
                    continue
                elif self.neighborhood(row, col).sum() > 0:
                    continue

                self.field[row, col] = self.food_types[food_type]
                food_count_type += 1
                food_count_total += 1

        if food_count_total < self.n_food:
            return False
        else:
            return True


    def _is_empty_location(self, row, col):

        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self):

        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows)
                col = self.np_random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self, observations):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.n_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[self.n_food * 3 + 2 * i] = -1
                obs[self.n_food * 3 + 2 * i + 1] = -1

            for i, p in enumerate(seen_players):
                obs[self.n_food * 3 + 2 * i] = p.position[0]
                obs[self.n_food * 3 + 2 * i + 1] = p.position[1]

            return obs

        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}

        return nobs, nreward, ndone, ninfo

    def reset(self):

        spawn_success = False
        while not spawn_success:
            self.field = np.zeros(self.field_size, np.int32)
            self.spawn_players()
            spawn_success = self.spawn_food()


        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ninfo = self._make_gym_obs(observations)
        return nobs

    def move_players(self, actions):
        self.current_step += 1

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)

        # and do movements for non colliding players
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

    def reward_players(self):

        # check if all players are adjacent to same food
        adj_food_locations = []
        players_fail_to_eat = False
        for player in self.players:
            adj_food_loc = self.adjacent_food_location(*player.position)
            if adj_food_loc is None:
                players_fail_to_eat = True
                break
            elif self.field[adj_food_loc] not in self.target_foods:
                players_fail_to_eat = True
                break
            else:
                adj_food_locations.append(adj_food_loc)

        if not players_fail_to_eat and adj_food_locations.count(adj_food_locations[0]) == len(adj_food_locations):
            self.field[adj_food_locations[0]] = 0
            for player in self.players:
                player.reward = 1 / self.n_target_foods
                player.score += player.reward
        else:
            for player in self.players:
                player.reward = 0


    def step(self, actions):

        self.move_players(actions)
        self.reward_players()

        self._game_over = (self.field.sum() == 0 or self._max_episode_steps <= self.current_step)
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]

        return self._make_gym_obs(observations)

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
