import gym
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation
from gym.wrappers import TransformObservation
import numpy as np
import retro
from skimage import transform


# 環境まわりの設定値
FRAME_SKIP = 4
SCREEN_SHAPE = 84
TIME_OUT = 3_000

# 終点
DONE_X = 10000

# ステージ
ACT = 1

# 開始位置のx
START_X = 80  # sonic1
# START_X = 96  # sonic2


def make_env(game='SonicTheHedgehog-Genesis',
             state='GreenHillZone.Act1'):
# def make_env(game='SonicTheHedgehog2-Genesis',
#              state='CasinoNightZone.Act1'):
    env = retro.make(game=game, state=state)
    env = SkipFrame(env, skip=FRAME_SKIP)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=SCREEN_SHAPE)
    env = TransformObservation(env, f=lambda x: x / 255)
    env = RewardWrapper(env, end_x=DONE_X)
    env = TimeOver(env, max_episode_steps=TIME_OUT)
    env = SonicActionWrapper(env)
    env = FrameStack(env, num_stack=FRAME_SKIP)

    return env


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        resize_obs *= 255
        return resize_obs.astype(np.uint8)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class SonicActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(SonicActionWrapper, self).__init__(env)
        # buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT',
        #            'C', 'Y', 'X', 'Z']
        buttons = env.buttons
        actions = []
        if env.gamename == 'SonicTheHedgehog-Genesis':
            actions = [['RIGHT'], ['LEFT'], ['A'], ['RIGHT', 'A'], ['LEFT', 'A']]
        elif env.gamename == 'SonicTheHedgehog2-Genesis':
            actions = [['RIGHT'], ['LEFT'], ['DOWN'], ['A'], ['RIGHT', 'A'], ['LEFT', 'A'], ['DOWN', 'A']]

        self._actions = []
        for act in actions:
            arr = np.array([False] * len(buttons))
            for btn in act:
                arr[buttons.index(btn)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, end_x=9_767):
        super().__init__(env)
        self._x = START_X
        self._end_x = end_x
        self._rings = 0
        self._score = 0
        # self._max_speed = 0

    def reset(self, **kwargs):
        self._x = START_X
        self._rings = 0
        self._score = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward = 0.0

        # # 右に行けたら正の報酬
        # if info['x'] > self._x:
        #     reward += 1.0
        #     self._x = info['x']

        # 右に行けたとき、移動距離が大きいほど報酬も大きく
        if info['x'] > self._x:
            reward += (info['x'] - self._x) / 100
            self._x = info['x']

        # リングが増えたら正の報酬、減ったら負の報酬
        if info['rings'] > self._rings:
            reward += (info['rings'] - self._rings) / 100
            self._rings = info['rings']
        elif info['rings'] < self._rings:
            reward -= 1.0
            self._rings = info['rings']

        # スコアが増えたら正の報酬
        if info['score'] > self._score:
            reward += (info['score'] - self._score) / 1000
            self._score = info['score']

        # エピソード終了条件
        if not done:
            done = True
            if info['lives'] < 3:
                reward = -1.0
            elif info['x'] > self._end_x or info['act'] > (ACT - 1):
                reward = 1.0
            else:
                done = False

        return state, reward, done, info


class TimeOver(gym.wrappers.TimeLimit):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
            reward = -1.0
        return observation, reward, done, info
