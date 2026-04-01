import torch
import numpy as np
import functools
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Union
from functools import partial
from einops import rearrange, repeat

def cat(list_of_tensors, dim=0):
    """
    Concatenate a list of tensors.
    """
    return functools.reduce(lambda x, y: torch.cat([x, y], dim=dim), list_of_tensors)


def catcat(list_of_lists_of_tensors, dim_outer=0, dim_inner=0):
    """
    Recursively concatenate a list of tensors.
    """
    return cat([cat(inner_list, dim_inner) for inner_list in list_of_lists_of_tensors], dim_outer)


def discounted(vals, gae_lambda:float):
    """
    Computes the discounted sum of values as used for the return in RL.
    """
    G = 0
    res = torch.zeros_like(vals)
    for i in reversed(range(vals.shape[-1])):  # （1-lambda) * (r_t + lambda * r_{t+1} + lambda **2 * r_{t+2} + ...)
        G = vals[..., i] + gae_lambda * G
        res[..., i] = G

    return res


def advantage(rewards, values, gamma:float, gae_lambda:float):
    """
    Computes the advantage of the given returns as compared to the estimated values, optionally using GAE.
    """
    device = rewards[0][0].device
    if gae_lambda == 0:
        returns = discounted(rewards, gamma)
        advantages = returns - values
    else:
        # Generalized Advantage Estimation (GAE) https://arxiv.org/abs/1506.02438
        # via https://github.com/inoryy/reaver/blob/master/reaver/agents/base/actor_critic.py
        values = torch.cat([values, torch.zeros((values.shape[0], 1, 1)).to(device)], dim=2)
        deltas = rewards + gamma * values[..., 1:] - values[..., :-1]
        advantages = discounted(deltas, gamma * gae_lambda)

    return advantages


class Buffer:

    """
    Utility class to gather a replay buffer. Computes returns and advantages over logged trajectories.
    """

    def __init__(self):
        self.count = 0  # number of trajectories
        self.batch_count = 0  # number of batches
        # environment
        self.batch_idx:List[List[torch.Tensor]] = []
        self.curr_extran:List[List[torch.Tensor]] = []
        self.rewards:List[List[torch.Tensor]] = []
        self.values:List[List[torch.Tensor]] = []
        # expert related
        self.expert_actions:List[List[torch.Tensor]] = []
        # student related
        self.actions:List[List[torch.Tensor]] = []
        self.action_logits:List[List[torch.Tensor]] = []
        self.action_logprobs:List[List[torch.Tensor]] = []
        self.cache = defaultdict(list)
        self.camera_info = defaultdict(list)

    def __len__(self):
        return self.count

    def start_trajectory(self):
        """
        Initializes the list into which all samples of a trajectory are gathered.
        """
        #
        self.count += 1
        if len(self.batch_idx) > 0:
            B = len(self.batch_idx[-1][-1])  # last step of the last trajectory
            self.batch_count += B
        self.batch_idx += [[]]
        self.curr_extran += [[]]
        self.rewards += [[]]
        self.values += [[]]
        self.expert_actions += [[]]
        self.actions += [[]]
        self.action_logits += [[]]
        self.action_logprobs += [[]]

    @torch.inference_mode()  # cache are output by frozen networks
    def log_traj_obs(self, cache:Dict[str, torch.Tensor], camera_info:Dict[str, Union[torch.Tensor,str,int]]):
        for key, value in cache.items():
            self.cache[key].append(value)
        for key, value in camera_info.items():
            if isinstance(value, torch.Tensor):
                self.camera_info[key].append(value)
            else:
                self.camera_info[key] = value

    @staticmethod
    def dict_merge(pretend_dict:Dict[str, Union[torch.Tensor,str,int]], append_dict:Dict[str, torch.Tensor]):
        res_dict = dict()
        for key, value in pretend_dict.items():
            if isinstance(value, torch.Tensor):
                res_dict[key] = torch.cat([pretend_dict[key], append_dict[key]], dim=0)
            else:
                res_dict[key] = value
        return res_dict
    
    @staticmethod
    def dict_reduce(dict_list:Dict[str, Union[List[torch.Tensor], str, int]]):
        res_dict = dict()
        for key, value in dict_list.items():
            if isinstance(value, list):
                res_dict[key] = torch.cat(value, dim=0)
            else:
                res_dict[key] = value  # str or int
        return res_dict
    
    @staticmethod
    def list_reduce(tensor_list:List[torch.Tensor]):
        return torch.cat(tensor_list, dim=0)
    
    @staticmethod
    def dict_index(dict_tensor:Dict[str, Union[torch.Tensor,str,int]], indices:torch.Tensor) -> Dict[str, Union[torch.Tensor,str,int]]:
        res_dict = dict()
        for key, value in dict_tensor.items():
            if isinstance(value, torch.Tensor):
                res_dict[key] = value[indices.to(value.device)]
            else:
                res_dict[key] = value
        return res_dict

    @torch.inference_mode()
    def log_step(self, curr_extran:torch.Tensor, state_value:torch.Tensor, reward:torch.Tensor,
                  expert_action:torch.Tensor, action:torch.Tensor, action_logit:torch.Tensor, action_logprob:torch.Tensor
                 ):
        """
        Logs a single step in a trajectory.
        """
        B = curr_extran.shape[0]
        batch_idx = self.batch_count + torch.arange(B).to(curr_extran.device)
        self.batch_idx[-1].append(batch_idx)
        self.curr_extran[-1].append(curr_extran)
        self.expert_actions[-1].append(expert_action)
        self.rewards[-1].append(reward)
        self.values[-1].append(state_value)
        self.actions[-1].append(action)
        self.action_logits[-1].append(action_logit)
        self.action_logprobs[-1].append(action_logprob)

    def get_returns_and_advantages(self, gamma:float, gae_lambda:float):
        """
        Computes the return and advantage per trajectory in the buffer.
        """
        returns = [discounted(cat(rewards, dim=-1), gamma).transpose(2, 1)
                   for rewards in self.rewards]  # per trajectory
        advantages = [advantage(cat(rewards, dim=-1), cat(values, dim=-1), gamma, gae_lambda).transpose(2, 1)
                      for rewards, values in zip(self.rewards, self.values)]
        return returns, advantages

    def get_samples(self, gamma:float, gae_lambda:float) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Gather all samples in the buffer for use in a torch.utils.data.TensorDataset.

        Args:
            gamma (float): decay factory for value computation
            gae_lambda (float): decay factor for gae computation

        Returns:
            _type_: [samples], cache, camera_info
        """
        samples = [self.batch_idx, self.curr_extran, self.expert_actions, self.values,
                   self.actions, self.action_logits, self.action_logprobs]
        samples += self.get_returns_and_advantages(gamma, gae_lambda)
        return [catcat(sample) for sample in samples], self.dict_reduce(self.cache), self.dict_reduce(self.camera_info)

    def clear(self):
        """
        Clears the buffer and all its trajectory lists.
        """
        self.count = 0
        self.batch_count = 0
        self.batch_idx.clear()
        self.rewards.clear()
        self.expert_actions.clear()
        self.curr_extran.clear()
        self.values.clear()
        self.actions.clear()
        self.action_logits.clear()
        self.action_logprobs.clear()
        self.cache.clear()
        self.camera_info.clear()


class GRPOBuffer:

    """
    Utility class to gather a replay buffer. Computes returns and advantages over logged trajectories.
    """

    def __init__(self, min_reward_delta:float, group_num:int):
        self.min_reward_delta = min_reward_delta
        self.group_num = group_num
        self.count = 0  # number of trajectories
        self.batch_count = 0  # number of batches
        # environment
        self.batch_idx:List[List[torch.Tensor]] = []
        self.curr_extran:List[List[torch.Tensor]] = []
        self.rewards:List[List[torch.Tensor]] = []
        # expert related
        self.expert_actions:List[List[torch.Tensor]] = []
        # student related
        self.actions:List[List[torch.Tensor]] = []
        self.action_logits:List[List[torch.Tensor]] = []
        self.action_logprobs:List[List[torch.Tensor]] = []
        self.cache = defaultdict(list)
        self.camera_info = defaultdict(list)

    def __len__(self):
        return self.count

    def start_trajectory(self):
        """
        Initializes the list into which all samples of a trajectory are gathered.
        """
        #
        self.count += 1
        if len(self.batch_idx) > 0:
            B = len(self.batch_idx[-1][-1])  # last step of the last trajectory
            self.batch_count += B
        self.batch_idx += [[]]
        self.curr_extran += [[]]
        self.rewards += [[]]
        self.expert_actions += [[]]
        self.actions += [[]]
        self.action_logits += [[]]
        self.action_logprobs += [[]]

    @torch.inference_mode()  # cache are output by frozen networks
    def log_traj_obs(self, cache:Dict[str, torch.Tensor], camera_info:Dict[str, Union[torch.Tensor,str,int]]):
        for key, value in cache.items():
            self.cache[key].append(value)
        for key, value in camera_info.items():
            if isinstance(value, torch.Tensor):
                self.camera_info[key].append(value)
            else:
                self.camera_info[key] = value
    @staticmethod
    def dict_merge(pretend_dict:Dict[str, Union[torch.Tensor,str,int]], append_dict:Dict[str, torch.Tensor]):
        res_dict = dict()
        for key, value in pretend_dict.items():
            if isinstance(value, torch.Tensor):
                res_dict[key] = torch.cat([pretend_dict[key], append_dict[key]], dim=0)
            else:
                res_dict[key] = value
        return res_dict
    
    @staticmethod
    def dict_reduce(dict_list:Dict[str, Union[List[torch.Tensor], str, int]]):
        res_dict = dict()
        for key, value in dict_list.items():
            if isinstance(value, list):
                res_dict[key] = torch.cat(value, dim=0)
            else:
                res_dict[key] = value  # str or int
        return res_dict
    
    @staticmethod
    def list_reduce(tensor_list:List[torch.Tensor]):
        return torch.cat(tensor_list, dim=0)
    
    @staticmethod
    def dict_index(dict_tensor:Dict[str, Union[torch.Tensor,str,int]], indices:torch.Tensor) -> Dict[str, Union[torch.Tensor,str,int]]:
        res_dict = dict()
        for key, value in dict_tensor.items():
            if isinstance(value, torch.Tensor):
                res_dict[key] = value[indices.to(value.device)]
            else:
                res_dict[key] = value
        return res_dict
    
    @torch.inference_mode()
    def log_step(self, curr_extran:torch.Tensor, reward:torch.Tensor,
                  expert_action:torch.Tensor, action:torch.Tensor, action_logit:torch.Tensor, action_logprob:torch.Tensor
                 ):
        """
        Logs a single step in a trajectory.
        """
        G = self.group_num
        B = reward.shape[0] / G
        reward_group = rearrange(reward, '(b g) -> b g',b=B, g=G)
        reward_group_min, reward_group_max = reward_group.min(dim=1), reward_group.max(dim=1)
        rev = reward_group_max - reward_group_min > self.min_reward_delta
        reward_std, reward_mean = torch.std_mean(reward_group[rev], dim=1, keepdim=True)
        normalized_reward = rearrange((reward_group[rev] - reward_mean) / reward_std, 'b g -> (b g)')
        B = curr_extran.shape[0]
        repeat_func = lambda y: repeat(y[rev], 'b ... -> (b g) ...', g=self.group_num)
        batch_idx = self.batch_count + torch.arange(B).to(curr_extran.device)
        self.batch_idx[-1].append(repeat_func(batch_idx))
        self.curr_extran[-1].append(repeat_func(curr_extran))
        self.expert_actions[-1].append(repeat_func(expert_action))
        self.rewards[-1].append(normalized_reward)
        self.actions[-1].append(repeat_func(action))
        self.action_logits[-1].append(repeat_func(action_logit))
        self.action_logprobs[-1].append(repeat_func(action_logprob))

    def get_returns(self, gamma:float):
        """
        Computes the return and advantage per trajectory in the buffer.
        """
        returns = [discounted(cat(rewards, dim=-1), gamma).transpose(2, 1)
                   for rewards in self.rewards]  # per trajectory
        return returns

    def get_samples(self, gamma:float) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Gather all samples in the buffer for use in a torch.utils.data.TensorDataset.

        Args:
            gamma (float): decay factory for value computation

        Returns:
            _type_: [samples], cache, camera_info
        """
        samples = [self.batch_idx, self.curr_extran, self.expert_actions,
                   self.actions, self.action_logits, self.action_logprobs, self.get_returns(gamma)]
        # returns = adavantage, returns is unnessary to use, since value function is removed
        return [catcat(sample) for sample in samples], self.dict_reduce(self.cache), self.dict_reduce(self.camera_info)

    def clear(self):
        """
        Clears the buffer and all its trajectory lists.
        """
        self.count = 0
        self.batch_count = 0
        self.batch_idx.clear()
        self.rewards.clear()
        self.expert_actions.clear()
        self.curr_extran.clear()
        self.actions.clear()
        self.action_logits.clear()
        self.action_logprobs.clear()
        self.cache.clear()
        self.camera_info.clear()

class SoIDBuffer:
    def __init__(self, min_ratio:float, max_ratio:float,
                min_len:int, max_len:int, sample_batch:int,
                dynamic_batch:bool, resample:bool, max_batch:int, epoch_interval_ratio:float):
        self.cnt = 0  # number of batches in one epoch
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_dist = None
        self.min_dist = None
        self.min_len = min_len
        self.max_len = max_len
        self.sample_batch = sample_batch
        self.dynamic_batch = dynamic_batch
        self.resample = resample
        self.max_batch = max_batch
        self.tmp_dist = list()
        self.new_samples = 0
        self.epoch_interval_ratio = epoch_interval_ratio  # 0~1
        self.cache_buffer = defaultdict(partial(deque, maxlen=max_len))
        self.init_extran_buffer = deque(maxlen=max_len)
        self.gt_extran_buffer = deque(maxlen=max_len)
        self.camera_info_buffer = defaultdict(partial(deque, maxlen=max_len))

    def calib_dist_threshold(self) -> bool:
        if len(self.tmp_dist) < self.min_len:
            print("cache is too small ({} < {}), cannot calibrate. Wait for the next epoch".format(len(self.tmp_dist), self.min_len))
            return False
        self.min_dist = np.quantile(self.tmp_dist, self.min_ratio)
        self.max_dist = np.quantile(self.tmp_dist, self.max_ratio)
        if self.dynamic_batch:
            self.sample_batch = int((len(self.init_extran_buffer) + self.new_samples * (1-self.epoch_interval_ratio)) / self.cnt)  # number of (samples + expected new samples) / number of batches
        self.sample_batch = min(self.sample_batch, self.max_batch)
        self.sample_batch = max(1, self.sample_batch)
        self.tmp_dist.clear()
        self.cnt = 0
        self.new_samples = 0
        return True
    
    @torch.inference_mode()
    def log_dist(self, final_dist:torch.Tensor):
        self.tmp_dist.extend(final_dist.tolist())

    @torch.inference_mode()
    def log_step(self, batch_i:int, total_batches:int, final_dist:torch.Tensor, init_extran:torch.Tensor, gt_extran:torch.Tensor, camera_info:Dict[str, torch.Tensor], cache:Dict[str, torch.Tensor]):
        assert self.min_dist is not None and self.max_dist is not None, "required to log_step() after calib_dist_threshold()"
        valid_rev = torch.logical_and(final_dist >= self.min_dist, final_dist <= self.max_dist)
        self.new_samples += valid_rev.sum().item()
        self.tmp_dist.extend(final_dist.tolist())
        self.init_extran_buffer.extend(torch.split(init_extran[valid_rev], 1 ,dim=0))  # (1, 4, 4)
        self.gt_extran_buffer.extend(torch.split(gt_extran[valid_rev], 1, dim=0))   # (1, 4, 4)
        for key, value in camera_info.items():
            if isinstance(value, torch.Tensor):
                self.camera_info_buffer[key].extend(torch.split(value[valid_rev.to(value.device)], 1, dim=0))  # (1, 1)
            else:
                self.camera_info_buffer[key] = value
        for key, value in cache.items():
            self.cache_buffer[key].extend(torch.split(value[valid_rev], 1, dim=0))
        self.cnt += 1
        if self.resample and len(self.init_extran_buffer) == self.max_len:
            self.resampler_buffer(batch_i / total_batches)

    def resampler_buffer(self, ratio:float):
        curr_len = len(self.init_extran_buffer)
        sample_num = int(curr_len * ratio)
        random_index = np.random.choice(curr_len, sample_num, replace=False)
        random_index.sort()
        for i, idx in enumerate(random_index):
            self.init_extran_buffer.append(self.init_extran_buffer[idx-i])
            self.gt_extran_buffer.append(self.gt_extran_buffer[idx-i])
            for key, value in self.camera_info_buffer.items():
                if isinstance(value, deque):
                    self.camera_info_buffer[key].append(self.camera_info_buffer[key][idx-i])
            for key, value in self.cache_buffer.items():
                self.cache_buffer[key].append(self.cache_buffer[key][idx-i])


    def clear(self):
        self.tmp_dist.clear()
        self.init_extran_buffer.clear()
        self.gt_extran_buffer.clear()
        self.cache_buffer.clear()
        self.camera_info_buffer.clear()
        self.cnt = 0
        self.new_samples = 0
    
    def sample(self) -> Tuple[Union[torch.Tensor,None], Union[torch.Tensor,None], Union[Dict[str, torch.Tensor],None], Union[Dict[str, torch.Tensor],None]]:
        """sample a batch from SoIDBuffer

        Returns:
            Tuple[Union[torch.Tensor,None], Union[torch.Tensor,None], Union[Dict[str, torch.Tensor],None], Union[Dict[str, torch.Tensor],None]]: return_init_extran, return_gt_extran, return_cache_buffer, return_camera_info
        """
        if not self.is_buffer_avilable():
            return None, None, None
        return_init_extran = torch.cat([self.init_extran_buffer.popleft() for _ in range(self.sample_batch)], dim=0)
        return_gt_extran = torch.cat([self.gt_extran_buffer.popleft() for _ in range(self.sample_batch)], dim=0)
        return_cache_buffer = dict()
        for key, value in self.cache_buffer.items():
            return_cache_buffer[key] = torch.cat([value.popleft() for _ in range(self.sample_batch)], dim=0)
        return_camera_info = dict()
        for key, value in self.camera_info_buffer.items():
            if isinstance(value, deque):
                return_camera_info[key] = torch.cat([value.popleft() for _ in range(self.sample_batch)], dim=0)
            else:
                return_camera_info[key] = value  # str or int
        return return_init_extran, return_gt_extran, return_cache_buffer, return_camera_info
    
    def is_buffer_avilable(self):
        return len(self.init_extran_buffer) >= self.sample_batch

    

    

    