import os

import random
import gymnasium as gym
from tqdm import tqdm
from GABRIL_utils.gaze_to_mask import GazeToMask


import matplotlib.pyplot as plt
import numpy as np
import torch
from GABRIL_utils.frame_writer import FrameWriter

from GABRIL_utils.gaze_utils import apply_gmd_dropout

MAX_EPISODES = {'Alien': 20, 'Asterix': 20, 'Assault': 20, 'Breakout': 20, 'ChopperCommand': 20,
                'DemonAttack': 20, 'Enduro': 20, 'Frostbite': 20, 'Freeway': 20, 'MsPacman': 20,
                'Phoenix': 20, 'Qbert': 20, 'RoadRunner': 20, 'Seaquest': 50, 'UpNDown': 20}

MAX_EPISODES_ATARI_HEAD = {'Seaquest': 20, 'MsPacman': 20, 'Enduro':20, 'Freeway': 10, 'Hero': 20, 'BankHeist': 20} # Atari_Head data

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(env, seed, datapath, conf_type, conf_randomness, stack, num_episodes=None, use_gaze=False, data_source='Our', gaze_mask_sigma=15.0, gaze_mask_coef=0.7):
    if num_episodes:
        if data_source == 'Atari_HEAD':
            assert num_episodes <= MAX_EPISODES_ATARI_HEAD[env], f'The number of available episodes is {MAX_EPISODES_ATARI_HEAD[env]}, but {num_episodes} is requested'
        else:
            assert num_episodes <= MAX_EPISODES[env], f'The number of available episodes is {MAX_EPISODES[env]}, but {num_episodes} is requested'
    
    # load the data
    path = os.path.join(datapath, env)
    file_name = f"/num_episodes_{MAX_EPISODES[env]}_fs4_human.pt"
    atari_head_file_name = f'Atari_Head_Data/{env}/ordinary.pt'
    loaded_obj = torch.load(atari_head_file_name) if data_source == 'Atari_HEAD' else torch.load(path + file_name, weights_only=False) 

    # u might be like why tf have that intermediary loaded_obj, but its helpful bc torch.load() is slow so u only wanna do it once
    
    obs = loaded_obj['observations']
    rewards = loaded_obj['episode-rewards']
    actions = loaded_obj['actions']
    episode_lengths = loaded_obj['steps']
    truncateds = loaded_obj['truncateds']
    terminateds = loaded_obj['terminateds']
    
    gaze_info = loaded_obj['gaze_information'] if use_gaze else None

    assert num_episodes <= episode_lengths.shape[0], "num_episodes should be less than total saved episodes"

    obs = torch.from_numpy(obs)
    actions = torch.from_numpy(actions)
    episode_lengths = torch.from_numpy(episode_lengths)
    truncateds = torch.from_numpy(truncateds)
    terminateds = torch.from_numpy(terminateds)

    if use_gaze:
        gaze_info = torch.from_numpy(gaze_info)

        g = gaze_info[:, :2]
        g[g < 0] = 0
        g[g > 1] = 1

        gaze_info[:, :2] = g


    short_memory_length = 20 
    stride = 2

    saliency_sigmas = [gaze_mask_sigma/(0.99**(short_memory_length - i)) for i in range(short_memory_length+1)]
    coeficients = [gaze_mask_coef**(short_memory_length - i) for i in range(short_memory_length+1)]
    coeficients += coeficients[::-1][1:]
    saliency_sigmas += saliency_sigmas[::-1][1:]

    MASK = GazeToMask(84, saliency_sigmas, coeficients=coeficients)

    episode_obs = []
    episode_actions = []
    
    episode_gaze = []

    for episode, _ in enumerate(episode_lengths):
        start = sum(episode_lengths[:episode])
        end = start + episode_lengths[episode]
        episode_obs.append(
            obs[start + episode:end + episode + 1])
    
        episode_actions.append(actions[start:end])
        if use_gaze:
            episode_gaze.append(gaze_info[start:end])
        assert terminateds[end - 1] or truncateds[end - 1]

    if num_episodes:
        episode_obs = episode_obs[:num_episodes]
        episode_actions = episode_actions[:num_episodes]
        if use_gaze:
            episode_gaze = episode_gaze[:num_episodes]

    episode_obs = [ep[:-1] for ep in episode_obs]
    if use_gaze:
        episode_saliency_gaze = [torch.stack(
                [MASK.find_bunch_of_maps(means=
                    ep[max(0, j - stride * short_memory_length):min(short_memory_length * stride + j + 1, len(ep)):stride],
                        offset_start=max(short_memory_length - j , 0))
                            for j in range(len(ep))], 0)
                                for ep in tqdm (episode_gaze)]
        episode_gaze_coordinates = [ep[:, :2] for ep in episode_gaze]
        
    else:
        episode_saliency_gaze =    [torch.zeros_like(episode_obs[i], dtype=torch.float32) for i in range(len(episode_obs))]
        episode_gaze_coordinates = [torch.zeros((len(episode_obs[i]), 2), dtype=torch.float32) for i in range(len(episode_obs))]
    
    # repeat the first frame for stack - 1 times
    episode_obs = [torch.cat([ep[0].unsqueeze(0)] * (stack - 1) + [ep]) for ep in episode_obs]
    episode_saliency_gaze = [torch.cat([ep[0].unsqueeze(0)] * (stack - 1) + [ep]) for ep in episode_saliency_gaze]

    for i, (ep_obs, ep_gaze) in enumerate(zip(episode_obs, episode_saliency_gaze)):
        new_episode_obs = []
        new_episode_saliency_gaze = []
        for s in range(stack):
            end = None if s == stack - 1 else s - stack + 1
            new_episode_obs.append(ep_obs[s:end])
            new_episode_saliency_gaze.append(ep_gaze[s:end])
        
        episode_obs[i] = torch.stack(new_episode_obs, dim=1)
        episode_saliency_gaze[i] = torch.stack(new_episode_saliency_gaze, dim=1)

    unique_actions = np.unique(actions)
    num_actions = unique_actions.max() + 1
    fw = FrameWriter((84,84), num_actions)

    if conf_type != 'normal':
        print("Building dataset by writing actions over the images..")

    rnd_generator = np.random.default_rng(seed)

    for i, ep in tqdm(enumerate(episode_obs), total=len(episode_obs)):
        new_obs = []
        for j in range(1, len(ep)):

            if conf_type == 'confounded':
                action_to_write = episode_actions[i][j - 1].item()
            elif conf_type == 'normal':
                action_to_write = None
            else:
                raise NotImplementedError(conf_type)

            if conf_type in ['confounded']:
                if rnd_generator.random() < conf_randomness:
                    action_to_write = rnd_generator.integers(num_actions)

            new_obs.append(fw.add_text_tensor_to_frame(ep[j].numpy(), action_to_write))

        episode_obs[i] = torch.from_numpy(np.stack(new_obs))

    episode_actions = [ep[1:] for ep in episode_actions]
    episode_saliency_gaze = [ep[1:] for ep in episode_saliency_gaze]
    episode_gaze_coordinates = [ep[1:] for ep in episode_gaze_coordinates]

    observations = torch.cat(episode_obs)
    actions = torch.cat(episode_actions)
    gaze_saliency_maps = torch.cat(episode_saliency_gaze)
    gaze_coordinates = torch.cat(episode_gaze_coordinates)

    return observations, actions, gaze_saliency_maps, gaze_coordinates

def evaluate(env, pre_actor, actor, model, device, args, verbose=False):
    model.eval()
    pre_actor.eval()
    actor.eval()
    if args.gaze_method == 'AGIL':
        args.encoder_agil.eval()

    num_actions = env.action_space.n
    fw = FrameWriter((84, 84), num_actions)

    scores = []
    for episode in tqdm(range(args.num_eval_episodes)):

        done = False
        episode_reward = 0
        step = 0

        stacked_obs, info = env.reset(seed=args.seed + 1000 * episode)
        rnd_generator = np.random.default_rng(args.seed + 1000 * episode)
        prev_action = 0

        while not done:

            stacked_obs = np.asarray(stacked_obs, dtype=np.uint8)

            if args.eval_type == 'confounded':
                action_index = prev_action

            else:
                action_index = None

            if args.eval_type in ['confounded']:
                if rnd_generator.random() < args.randomness:
                    action_index = rnd_generator.integers(num_actions)

            stacked_obs = fw.add_text_tensor_to_frame(stacked_obs, action_index)

            _ = env.get_wrapper_attr('render_')(gaze=None,
                                                      record_frame=episode < 2 if args.eval_record else False)

            with torch.no_grad():
                stacked_obs = torch.as_tensor(stacked_obs, device=device, dtype=torch.float32).unsqueeze(0) / 255.0
                if args.gaze_method in ['ViSaRL', 'Mask', 'AGIL'] or args.dp_method in ['IGMD', 'GMD']: # models that need gaze prediction
                    g = args.gaze_predictor(stacked_obs)
                    g[g < 0] = 0
                    g[g > 1] = 1
                
                if args.gaze_method == 'ViSaRL':
                    stacked_obs = torch.cat([stacked_obs, g], dim=1)
                elif args.gaze_method == 'Mask':
                    stacked_obs = stacked_obs * g
                
                dropout_mask = None
                if args.dp_method == 'IGMD':
                    dropout_mask = g[:,-1:]

                z = model(stacked_obs, dropout_mask=dropout_mask)
                
                if args.gaze_method == 'AGIL':
                    z = (z + args.encoder_agil(stacked_obs * g)) / 2
                
                if args.dp_method == 'GMD':
                    z = apply_gmd_dropout(z, g[:,-1:], test_mode=True)

                z = pre_actor(z)
                action = actor(z).argmax(1)[0].cpu().item()

            stacked_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            prev_action = action

            episode_reward += reward
            step += 1
            
            if step > 5000:
                done = True

        scores.append(episode_reward)
        if verbose:
            print(
                f"Episode {episode} reward: {episode_reward}(mean: {np.mean(scores)}, std: {np.std(scores)}), steps: {step}")

    if args.eval_record:
        print(env.get_wrapper_attr('save_video')(f'vids/{args.add_path}.mp4'))
    return np.mean(scores), np.std(scores)

def plot_gaze_and_obs(gaze, obs, save_path=None):
    # Create data for the plots
    y1 = gaze
    
    y2 = obs
    if obs.dtype == torch.uint8:
        y2 = obs.to(torch.float32)/255
    
    y3 = (y1 * y2).to(torch.float32)

    if len(y3.shape) == 3:
        if y3.shape[0] == 3:
            y3 = y3.permute(1, 2, 0)
        elif y3.shape[0] == 1:
            y3 = y3[0]

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Plot 1
    ax1.imshow(y1, cmap='gray', vmax=1.0, vmin=0.0)
    ax1.set_title('pure gaze')

    # Plot 2
    ax2.imshow(y2, cmap='gray', vmax=1.0, vmin=0.0)
    ax2.set_title('pure obs')

    # Plot 3
    ax3.imshow(y3, cmap='gray', vmax=1.0, vmin=0.0)
    ax3.set_title('merged gaze and obs')
    
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    
    # Show the plots
    plt.show()

if __name__ == '__main__':
    observations, actions, gaze_saliency_maps, gaze_coordinates =   load_dataset(env='Seaquest', 
                                                                    seed=1, 
                                                                    datapath='dataset', 
                                                                    conf_type='normal', 
                                                                    conf_randomness=0.0, 
                                                                    stack=1, 
                                                                    num_episodes=50, 
                                                                    use_gaze=True)