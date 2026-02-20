from typing import Dict, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import pygame
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from collections import deque
from gymnasium.wrappers.frame_stack import LazyFrames

from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn

import cv2

class StickyActionEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sticky action.

    Paper: https://arxiv.org/abs/1709.06009
    Official implementation: https://github.com/mgbellemare/Arcade-Learning-Environment

    :param env: Environment to wrap
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self._sticky_action = 0  # NOOP
        return self.env.reset(**kwargs)

    def step(self, action: int) -> AtariStepReturn:
        if self.np_random.random() >= self.action_repeat_probability:
            self._sticky_action = action
        return self.env.step(self._sticky_action)



class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info: Dict = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info



class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}



class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> AtariStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> AtariResetReturn:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info



class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int) -> AtariStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if i < self._skip - 1:
                self.env.get_wrapper_attr('dummy_render')()
            
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info
    


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: SupportsFloat) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))
    


class WarpFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :]
    

#   Wrapper which renders the game in a determined size
class RenderWrapper(ObservationWrapper):
    def __init__(self, env, width=1920, height=1080, render_mode='human', frame_skip=1, gaze_circle_size=15):
        super().__init__(env)
        self.width = width
        self.height = height
        self.mode = render_mode
        self.frame_skip = frame_skip
        self.gaze_circle_size = gaze_circle_size

        self.obs_history = []

        if render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))

    def observation(self, observation):
        return observation
    
    def fill_obs_history(self, obs):
        if len(self.obs_history) < self.frame_skip:
            self.obs_history.append(obs)
        else:
            self.obs_history = self.obs_history[1:]
            self.obs_history.append(obs)
    
    def dummy_render(self):
        obs = self.env.get_wrapper_attr('render')()
        self.fill_obs_history(obs)
    
    def get_max_history(self):
        ob = np.stack(self.obs_history, 0)
        return np.max(ob, 0)

    def render_(self, gaze=None, record_frame = False): # gaze is a tuple of (x, y)

        if (not record_frame) and self.mode == 'rgb_array':
            return

        obs = self.env.get_wrapper_attr('render')()
        self.fill_obs_history(obs)

        obs = self.get_max_history()


        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if not (gaze is None):
            if not isinstance (gaze, np.ndarray):
                gaze = np.array(gaze)
            
            gaze[0] = gaze[0] * self.width
            gaze[1] = gaze[1] * self.height
            # Add gaze marker here
            pix_square_size = self.gaze_circle_size / 1920 * self.width
            # Add circle here
            cv2.circle(obs, (int(gaze[0]), int(gaze[1])), int(pix_square_size), (190, 0, 190), -1)

        if self.mode == 'human':
            modified_observation = np.transpose(obs, (1, 0, 2))
            surfarray = pygame.surfarray.make_surface(modified_observation)
            surfarray = pygame.transform.scale(surfarray, (self.width, self.height))

            self.window.blit(surfarray, (0,0))
            pygame.event.pump()
            pygame.display.update()

        return obs

class VideoRecorderWrapper(gym.Wrapper):
    def __init__(self, env, width=1440, height=900, fps = 60):
        self.img_list = []     #   Keeps a record of saveable frames
        self.width = width
        self.height = height
        self.fps = fps
        super().__init__(env)

    def render_(self, gaze=None, record_frame=False):
        img = self.env.get_wrapper_attr('render_')(gaze, record_frame)
        if record_frame:
            self.img_list.append(img)
        return img

    def save_video(self, save_path=f'vids/vid.mp4'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            # check that moviepy is now installed
            import moviepy
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        except ImportError:
            raise ImportError("Video recording requires the moviepy library")

        clip = ImageSequenceClip(self.img_list, fps=self.fps)
        clip.write_videofile(save_path, codec='libx264')
        
        self.img_list = []

        return 'Video saved!'

class RamManagerWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env: gym.Env, ram_env:gym.Env) -> None:
        super().__init__(env)
        self.ram_env = ram_env

    def reset(self, **kwargs) -> AtariResetReturn:
        ram_observation, _ = self.ram_env.reset(**kwargs)
        observation, info = self.env.reset(**kwargs)
        info['ram'] = ram_observation
        return observation, info

    def step(self, action: int) -> AtariStepReturn:
        ram_observation, ram_reward, ram_terminated, ram_truncated, ram_info = self.ram_env.step(action)
        observation, reward, terminated, truncated, info = self.env.step(action)

        assert ram_reward == reward, 'Rewards from ram-env and main-env are not the same!'
        assert ram_terminated == terminated, 'Terminations of ram-env and main-env are not the same!'
        assert ram_truncated == truncated, 'Tryncations of ram-env and main-env are not the same!'
        
        info['ram'] = ram_observation
        return observation, reward, terminated, truncated, info
    
    
class FrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
        - After :meth:`reset` is called, the frame buffer will be filled with the initial observation.
          I.e. the observation returned by :meth:`reset` will consist of `num_stack` many identical frames.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStack
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        lz4_compress: bool = False,
        save_ram: bool = False
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, num_stack=num_stack, lz4_compress=lz4_compress
        )
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
        self.save_ram = save_ram
        if self.save_ram:
            self.ram_data = []

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)
    
    def info_update(self, info):
        self.ram_data.append(info['ram'])
        self.ram_data = self.ram_data[1:]
        info['ram'] = np.stack(self.ram_data, 0)
        return info

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        if self.save_ram:
            info = self.info_update(info=info)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)

        [self.frames.append(obs) for _ in range(self.num_stack)]
        if self.save_ram:
            self.ram_data = []
            [self.ram_data.append(info['ram']) for _ in range(self.num_stack)]
            info = self.info_update(info=info)
        return self.observation(None), info
    

class AtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        obs_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
        screen_render_width: int = 1920,
        screen_render_height: int = 1080,
        render_mode: str = 'rgb_array',
        num_stack: int = 1,
        start_fire: bool = False,
        ram_env = None
    ) -> None:
        if not (ram_env is None):
            env = RamManagerWrapper(env, ram_env=ram_env)
        env = RenderWrapper(env, width=screen_render_width, height=screen_render_height, render_mode=render_mode, frame_skip=frame_skip)
        env = VideoRecorderWrapper(env, width=screen_render_width, height=screen_render_height, fps=60 / frame_skip)
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if  start_fire and "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        env = WarpFrame(env, width=obs_size, height=obs_size)
        if clip_reward:
            env = ClipRewardEnv(env)
        env = FrameStack(env=env, num_stack=num_stack, save_ram=not(ram_env is None))

        super().__init__(env)


def create_env(
        env_name: str = 'Seaquest',
        noop_max: int = 30,
        frame_skip: int = 1,
        obs_size: int = 84,
        terminal_on_life_loss: bool = False,
        clip_reward: bool = False,
        action_repeat_probability: float = 0.0,
        screen_render_width: int = 1920,
        screen_render_height: int = 1080,
        render_mode: str = 'rgb_array',
        num_stack: int = 1,
        start_fire: bool = False,
        ram_info: bool = False
        ):
    env = gym.make(f'{env_name}NoFrameskip-v4', render_mode='rgb_array')
    ram_env = gym.make(f'{env_name}-ramNoFrameskip-v4') if ram_info else None

    env.metadata['render_fps'] = 20
    
    return AtariWrapper(env, noop_max=noop_max, frame_skip=frame_skip, obs_size=obs_size,
                        terminal_on_life_loss=terminal_on_life_loss, clip_reward=clip_reward,
                        action_repeat_probability=action_repeat_probability, start_fire= start_fire,
                        screen_render_width=screen_render_width,
                        screen_render_height=screen_render_height,
                        render_mode=render_mode, num_stack=num_stack, ram_env=ram_env)