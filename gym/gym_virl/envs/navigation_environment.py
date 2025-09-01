import os
from typing import Optional

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from itertools import permutations, product, chain, zip_longest
from fractions import Fraction as F
# from data_collection.verifier import step_rewards
from utils_rl import step_rewards_virl
from utils_general import re_match, robust_str_to_list, parse_navigation_string, parse_direction_string
from PIL import Image, ImageDraw, ImageFont
from utils_rl import REWARD_FN_VIRL
import json
from virl.platform.platform import Platform
from virl.utils import geocode_utils, pipeline
import os
from pathlib import Path
def load_json_lines(filename):
    data = []
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
    return data
def save_json_line(filename, new_entry):
    # Load existing data
    existing_data = load_json_lines(filename)
    
    # Check if entry already exists
    if new_entry not in existing_data:
        # Append new entry
        with open(filename, 'a') as file:
            if os.path.getsize(filename) > 0:
                file.write('\n')
            json.dump(new_entry, file)
        return True
    return False
class NavigationEnvironment(gym.Env):
    """
    Env description
    """
    
    def __init__(self, 
        route_info_path: str, 
        relocation: bool=False,
        drop_rate: float=0.0,
        straight_line_length: int=2,
        resolution: int=1200,
        verify_iter: int=0,
        platform_cfg: dict=None,
        navigator_cfg: dict=None,
        platform_save_dir: str=None,
        language_only: bool=True,
        absolute_action: bool=True
        ):
        """
        absolute_action False if ['forward()', 'turn_direction(left)', 'turn_direction(right)', 'stop()']
        """
        self.absolute_action = absolute_action 
        self.language_only = language_only
        self.relocation = relocation
        self.drop_rate = drop_rate
        self.straight_line_length = straight_line_length
        self.resolution = resolution
        self.verify_iter = verify_iter
        self.route_info_path = route_info_path
        self.route_list = json.load(open(route_info_path))[0]
        self.route_num = json.load(open(route_info_path))[1]
        # not used but needed for compatibility
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(2*resolution + 5, 2*resolution + 5, 3), dtype=np.uint8)
        os.makedirs(platform_save_dir, exist_ok=True)
        self.platform = Platform(platform_cfg, Path(platform_save_dir))
        self.cfg = navigator_cfg
        self.language_only = language_only
        
        self.orientation_set = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
        self.orientation_heading = [0, 45, 90, 135, 180, 225, 270, 315]
        # reset 10 times
        # for i in range(10):
        #     self.reset()
        #     self._simulate_one_trajectory_in_rail()
        # assert False
        # for i in range(5):
        #     self.reset()
        #     self._simulate_one_trajectory_in_rail()
        #     polyline = geocode_utils.encode_polyline(self.traj_list)
        #     print(self.str_instruction)
        #     pipeline.draw_planned_route(
        #         self.platform, polyline, input_way_points=None,
        #         path=f"./output/debug_{i}.html",
        #         file_template=self.cfg.FILE_TEMPLATE
        #     )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.verify_info = None
        
        if options is not None:
            self.route_info = self.route_list[options['route_idx']]
        else:
            self.route_info = random.choice(self.route_list)
        # self.route_info = self.route_list[0]
        # print(self.route['milestone_info'])
        start_place = self.route_info['start_place']
        # print(start_place)
        start_position = start_place['relocated_geocode']
        # print("Navigation start at: ", start_position)
        if self.relocation:
            self.current_geocode = pipeline.position_to_geocode(self.platform, start_position)
        else:
            self.current_geocode = start_position
        self.current_heading = self.route_info.get('init_heading', 0)
        if self.current_heading == 0:
            # random sample an int from 0 to 359
            self.current_heading = random.randint(0, 359)
        self.target_heading = self.current_heading
        self.landmark_list = self.route_info['route_results']['landmark_list'] + [self.route_info['dest_place']]
        
        self.action_list = []
        self.observation_list = []
        self.traj_list = [self.current_geocode]
        # self.platform.initialize_mover(self.current_geocode)

        self.meta_info = {}

        info = {}
        
        
        self.str_instruction, self.list_instruction = self._parse_instruction_and_rail()
        self.instruction_idx = 0
        self.landmark_idx = 0 # count the number of landmarks or intersections that have been visited
        self.step_cnt = 0
        self.remaining_step = self.verify_iter
        current_obs = self._get_observation_rail()
        self.observation_list.append(current_obs['oracle_observation'])
        self.is_success = True
        # self._simulate_one_trajectory_in_rail()
        
        # info['instruction'] = self.str_instruction
        # info['list_instruction'] = self.list_instruction
        # info['instruction_idx'] = self.instruction_idx # current sentence to be executed
        # info['gt_action'] = self._parse_gt_action()
        # info['obs_action_sequence'] = self.get_observation_action_sequence(self.action_list, self.observation_list)
        # # print(self.observation_list[-1])
        # # print(info)
        info = {
            'global_instruction': self.str_instruction,
            'current_instruction': self.gt_rail_info[0]['instruction'],
            'instruction_idx': self.instruction_idx,
            'gt_action': self.gt_rail_info[0]['gt_action'],
            'current_obs': current_obs['oracle_observation'],
            'obs_act_seq': self.get_observation_action_sequence(self.action_list, self.observation_list),
            'Verify Info': None, 
            'is_success': self.is_success
        }
        return current_obs['visual_observation'], info
    
    def _simulate_one_trajectory_in_rail(self): 
        # cnt = 0
        while True:
            rail_info = self.gt_rail_info[self.step_cnt]
            action = rail_info['gt_action']
            if action == "stop()":
                break
            self.move_on_rail(action)

    
    def _get_observation_rail(self):
        intersection_obs = self.gt_rail_info[self.step_cnt]['intersection_observation']
        # if stop() then there is no intersection observation
        if self.gt_rail_info[self.step_cnt]['gt_action'] == "stop()":
            intersection_obs = ""
        landmark_obs = self.gt_rail_info[self.step_cnt]['observation']
        self.current_heading = self.gt_rail_info[self.step_cnt]['heading']
        # self.observation_list.append(landmark_obs + ';' + intersection_obs)
        obs_dict = {
            "oracle_observation": landmark_obs + '; ' + intersection_obs,
            "visual_observation": None if self.language_only else self._get_visual_observation()
        }
        
        return obs_dict

    @staticmethod
    def _list2tup(lst):
        return (lst[0], lst[1])
    
    def _get_visual_observation(self):
        image_list = self.platform.get_all_streetview_from_geocode(
            self._list2tup(self.current_geocode), cur_heading=self.current_heading
        )
        
        # cat 4 images into a 2x2 grid, split each image with a black line of 10 pixels
        image_list = [image.image for image in image_list]
        line_width = 5
        canvas = Image.new('RGB', (self.resolution * 2 + line_width, self.resolution * 2 + line_width), (0, 0, 0))
        
        for i, image in enumerate(image_list):
            x = i % 2
            y = i // 2
            image = image.resize((self.resolution, self.resolution))
            # Adjust positioning to account for the line width
            pos_x = x * (self.resolution + line_width)
            pos_y = y * (self.resolution + line_width)
            canvas.paste(image, (pos_x, pos_y))
            
        np_canvas = np.array(canvas)
        return np_canvas
    
    def move_on_rail(self, action):
        self.step_cnt += 1
        self.remaining_step = self.verify_iter # reset the remaining step since the agent has moved
        self.current_geocode = self.gt_rail_info[self.step_cnt]['geocode']
        self.traj_list.append(self.current_geocode)
        self.action_list.append(action)
        self.observation_list.append(self._get_observation_rail()['oracle_observation'])
    
    def get_suceess(self):
        return self.is_success
    
    def _terminate_step(self, reward, info_key, is_truncated=False):
        self.is_success = False
        if self.gt_rail_info[self.step_cnt]['gt_action'] == "stop()":
            # failed at stopping position with max trials
            return None, reward, True, is_truncated, {
                'global_instruction': self.str_instruction,
                'current_instruction': self.gt_rail_info[self.step_cnt]['instruction'],
                'instruction_idx': self.gt_rail_info[self.step_cnt]['instruction_idx'],
                'gt_action': self.gt_rail_info[self.step_cnt]['gt_action'],
                'Verify Info': "Exceed maximum verify steps",
                'current_obs': self._get_observation_rail()['oracle_observation'],
                'obs_act_seq': self.get_observation_action_sequence(self.action_list, self.observation_list),
                'is_success': self.is_success
            }
        # failed at non-stopping position with max trials, move to the next position
        self.step_cnt += 1
        current_obs = self._get_observation_rail()
        info = {
            'global_instruction': self.str_instruction,
            'current_instruction': self.gt_rail_info[self.step_cnt]['instruction'],
            'instruction_idx': self.gt_rail_info[self.step_cnt]['instruction_idx'],
            'gt_action': self.gt_rail_info[self.step_cnt]['gt_action'],
            'Verify Info': "Exceed maximum verify steps",
            'current_obs': current_obs['oracle_observation'],
            'obs_act_seq': self.get_observation_action_sequence(self.action_list, self.observation_list),
            'is_success': self.is_success
        }
        return current_obs['visual_observation'], reward, not is_truncated, is_truncated, info
        
    def step(self, action):
        terminated, reward, info = False, 0, {}
        truncated = False
        
        self.remaining_step -= 1
        if self.remaining_step == -1:
            
            return self._terminate_step(-1, 'step_limit_reached', is_truncated=True)
            
        gt_action = self.gt_rail_info[self.step_cnt]['gt_action']
        gt_obs = self.gt_rail_info[self.step_cnt]['observation']
        gt_instruction = self.gt_rail_info[self.step_cnt]['instruction']
        reward, verify_message = step_rewards_virl(action, gt_action, gt_obs, gt_instruction)
        # print("verify_message: ", verify_message)
        # print("reward: ", reward)
        # print("gt_action: ", gt_action)
        # print(action)
        # assert False
        output_action = re_match(action, 'action')
        if reward == max(REWARD_FN_VIRL.values()):
            if gt_action == "stop()":
                terminated = True
            else:
                self.move_on_rail(gt_action)
            
        current_obs = self._get_observation_rail()
        info = {
            'global_instruction': self.str_instruction,
            'current_instruction': self.gt_rail_info[self.step_cnt]['instruction'],
            'instruction_idx': self.gt_rail_info[self.step_cnt]['instruction_idx'],
            'gt_action': self.gt_rail_info[self.step_cnt]['gt_action'],
            'Verify Info': verify_message,
            'current_obs': current_obs['oracle_observation'],
            'obs_act_seq': self.get_observation_action_sequence(self.action_list, self.observation_list), 
            'is_success': self.is_success
        }
        # assert False
        
        return current_obs['visual_observation'], reward, terminated, truncated, info
    
    
    def _parse_gt_action(self):
        instruction_to_be_executed = self.list_instruction[self.instruction_idx]
        # if in the starting position
        # 
        if self.absolute_action:
            if self.instruction_idx == 0:
                action = f"turn_direction({parse_direction_string(self.list_instruction[self.instruction_idx])})"
            else:
                if "Turn direction" in instruction_to_be_executed:
                    # check distance
                    dist = geocode_utils.calculate_distance_from_geocode(self.current_geocode, self.route_info['route_results']['geocode_list'][self.landmark_idx])
                    # print(f"Distance to next intersection: {dist}")
                    if dist < self.cfg.LANDMARK_DETECT.INTERSECTION_VALID_RADIUS and self.meta_info['intersection']:
                        action = f"turn_direction({parse_direction_string(self.list_instruction[self.instruction_idx])})"
                    else:
                        action = "forward()"
                else:
                    action = "forward()"
        else:
            if 'left' in instruction_to_be_executed:
                action = 'turn_direction(left)'
            elif 'right' in instruction_to_be_executed:
                action = 'turn_direction(right)'
            else:
                action = 'forward()'
        return action
    
    def _parse_instruction_and_rail(self):
        intersection_dict_list = parse_navigation_string(self.route_info['milestone_info'])
        # print(self.route_info['milestone_info'])
        parsed_instruction = []
        intersection_actions = []
        relative_actions = []
        heading_list = [self.current_heading]
        landmark_list = []
        prev_heading = self.current_heading
        for idx, intersection_dict in enumerate(intersection_dict_list):
            landmark = intersection_dict.get('landmarks', "No landmark nearby")
            landmark_list.append(landmark)
            to_next_intersection_heading = intersection_dict.get('to_next_intersection_heading', None) # like  '177 (south)'
            # '177 (south)' -> 'south'
            next_heading = to_next_intersection_heading.split('(')[1].split(')')[0] if to_next_intersection_heading is not None else None
            # '177 (south)' -> 177
            next_heading_deg = int(to_next_intersection_heading.split('(')[0].strip()) if to_next_intersection_heading is not None else None
            heading_list.append(next_heading_deg)
            # non-starting position has relative heading
            relative_heading_deg = next_heading_deg - prev_heading
            if relative_heading_deg < -45:
                relative_heading = 'left'
            elif relative_heading_deg > 45:
                relative_heading = 'right'
            elif -45 <= relative_heading_deg <= 0:
                relative_heading = 'slightly left'
            elif 0 <= relative_heading_deg <= 45:
                relative_heading = 'slightly right'
                    
            # <b>left</b> -> left
            prev_heading = next_heading_deg
            intersection_actions.append('turn_direction(' + next_heading + ')')
            relative_actions.append('turn_direction(' + relative_heading + ')')
            if idx == len(intersection_dict_list) - 1:
                parsed_instruction.append("Turn " + relative_heading + " to face " + next_heading + ".")
                parsed_instruction.append("Move forward until you reach destination where " + landmark + ".")
            else:
                first_msg = "First, turn" if idx == 0 else "Turn"
                if landmark == "No landmark nearby":
                    parsed_instruction.append(f"{first_msg} " + relative_heading + " to face " + next_heading + ".")
                    parsed_instruction.append("Move forward until you reach next intersection.")
                else:
                    parsed_instruction.append(f"{first_msg} " + relative_heading + " to face " + next_heading + ".")
                    parsed_instruction.append("Move forward until you reach next intersection where " + landmark + ".")
        print(f"Parsed instruction: {parsed_instruction}")
        # assert False
        """
        Parse the instruction like:
        1. instruction 1;
        2. instruction 2;
        ...
        """
        str_instruction = ""
        for idx, instr in enumerate(parsed_instruction):
            str_instruction += f"{idx + 1}. {instr}\n"
        
        """
        On the side, we parse a geocode rail to support a sparse navigation points
        """
        
        self.gt_rail_info = [] # list of dicts
        landmark_list.insert(0, "No landmarks nearby")
        intersection_actions.append("stop()")
        relative_actions.append("stop()")
        geocode_prev = self.current_geocode
        instruction_idx = 0
        for idx, geocode in enumerate([self.current_geocode]+ self.route_info['route_results']['geocode_list']):
            distance = geocode_utils.calculate_distance_from_geocode(geocode_prev, geocode)
            if self.relocation:
                relocated_geocode_intersection, panoid = self.platform.relocate_geocode_by_source(geocode, source='outdoor', direct_return = True)
                geocode = relocated_geocode_intersection
                # save pano id in data/panoid.json
                    
            if distance > 5:
                # if distance between two geocodes is larger than 5 meters, we interpolate the geocodes in between, s.t. the distance between two geocodes is less than 10 meters
                # n_of_points = min(int(distance / 5), self.straight_line_length)
                n_of_points = self.straight_line_length
                for i in range(n_of_points):
                    # drop the points with probablity of self.drop_rate, between 0 and 1
                    if random.random() < self.drop_rate:
                        continue
                    interpolated_geocode = self.interpolate_geocodes(geocode_prev, geocode, i+1, n_of_points)
                    if self.relocation:
                        try:
                            relocated_geocode, panoid = self.platform.relocate_geocode_by_source(interpolated_geocode, source='outdoor', direct_return = True)
                            assert relocated_geocode is not None
                        except:
                            continue
                    else:
                        relocated_geocode = interpolated_geocode
                    # add the geocode if it's not too close to the previous one
                    if self.relocation:
                        if (geocode_utils.calculate_distance_from_geocode(geocode_prev, relocated_geocode) > 8 and 
                           geocode_utils.calculate_distance_from_geocode(geocode, relocated_geocode) > 8):
                            self.gt_rail_info.append(
                                {
                                    'geocode': relocated_geocode,
                                    'heading': heading_list[idx],
                                    'gt_action': "forward()",
                                    'relative_action': "forward()",
                                    'observation': "No landmarks nearby",
                                    'intersection_observation': "",
                                    'instruction': parsed_instruction[instruction_idx],
                                    'instruction_idx': instruction_idx + 1, # count from 1
                                }
                            )
                    else:
                        self.gt_rail_info.append(
                            {
                                'geocode': relocated_geocode,
                                'heading': heading_list[idx],
                                'gt_action': "forward()",
                                'observation': "No landmarks nearby",
                                'intersection_observation': "",
                                'instruction': parsed_instruction[instruction_idx],
                                'instruction_idx': instruction_idx + 1, # count from 1
                            }
                        ) 
                instruction_idx += 1
            # stop() and last forward() share the same instruction
            if intersection_actions[idx] == 'stop()':
                instruction_idx -= 1
            self.gt_rail_info.append(
                {
                    'geocode': geocode,
                    'heading': heading_list[idx],
                    'gt_action': intersection_actions[idx] if self.absolute_action else relative_actions[idx],
                    'relative_action': relative_actions[idx],
                    'observation': landmark_list[idx],
                    'intersection_observation': "You observe an intersection" if (idx != 0) else "",
                    'instruction': parsed_instruction[instruction_idx],
                    'instruction_idx': instruction_idx + 1, # count from 1
                }
            )
            instruction_idx += 1
            geocode_prev = geocode
        
        
        # polyline = geocode_utils.encode_polyline([info['geocode'] for info in self.gt_rail_info])
        # pipeline.draw_planned_route(
        #     self.platform, polyline, input_way_points=None,
        #     path="./output/debug_rail.html",
        #     file_template=self.cfg.FILE_TEMPLATE
        # )
        
        return str_instruction, parsed_instruction
    
    
    # def _relocate(self, geocode):
    #     if self.relocation:
    #         relocated_geocode, panoid = self.platform.relocate_geocode_by_source(geocode, source='outdoor')
    #         # save the panoid in
    #     else:
    #         relocated_geocode = geocode
    #         panoid = None
    #     return relocated_geocode, panoid
    
    def _get_observation(self, info):
        intersection_obs = self.get_intersection_observation(info)
        landmark_obs = self.get_landmark_observation_oracle(info)
        
        return intersection_obs, landmark_obs
        

    @staticmethod
    def interpolate_geocodes(geocode_prev, geocode_next, i, n):
        # interpolate between two tuples
        tup_to_ret = (geocode_prev[0] * (1 - i / n) + geocode_next[0] * (i / n), geocode_prev[1] * (1 - i / n) + geocode_next[1] * (i / n))
        return tup_to_ret

    def get_observation_action_sequence(self, action_list, observation_list): 
        previous_observations = ""
        for i, (action, observation) in enumerate(zip(action_list, observation_list)):
            previous_observations += f"O_{i + 1}: {observation}\n"
            previous_observations += f"A_{i + 1}: {action}\n"
        
        if len(observation_list) > len(action_list):
            if not self.language_only:
                inter = "You observe an intersection" if "You observe an intersection" in observation_list[-1] else ""
                previous_observations += f"O_{len(observation_list)}: You observe an image of 4 views; {inter}\n"
                previous_observations += f"A_{len(observation_list)}: \n"
            else:
                previous_observations += f"O_{len(observation_list)}: {observation_list[-1]}\n"
                previous_observations += f"A_{len(observation_list)}: \n"
        return previous_observations

    def get_intersection_observation(self, info_dict):
        # since current intersection observation should be oracle, we need to avoid provide too much noise
        intersection_valid_here = True
        if self.route_info is not None and len(self.route_info['route_results']['geocode_list']) > 1:
            intersect_list = self.route_info['route_results']['geocode_list'][:-1]
            distance = geocode_utils.cal_distance_between_two_position_list([self.current_geocode], intersect_list)[0]
            min_dist = np.min(distance)
            if min_dist > self.cfg.LANDMARK_DETECT.INTERSECTION_VALID_RADIUS:
                print(f'>>> VisionLanguageNavigator: min_dist: {min_dist}')
                intersection_valid_here = False
                
        print(f'intersection_valid_here: {intersection_valid_here}')
        
        # for previous detect a landmark, and the agent decide to turn direction.
        # we need to check whether the agent is at the intersection
        if len(self.action_list) > 0 and 'turn_direction' in self.action_list[-1] and \
                'No landmarks nearby' not in self.observation_list[-1]:
            intersection_valid_here = True
        
        heading_list = self.platform.mover.get_all_suitable_heading_to_path_vln(
            self.current_geocode, radius_query=intersection_valid_here
        )
        info_dict['heading_list'] = heading_list
        
        if len(heading_list) > 2 and intersection_valid_here:
            observation = f' There are {len(heading_list)}-way intersections.'
            info_dict['intersection'] = True
        else:
            observation = ""
            info_dict['intersection'] = False

        return observation

    def get_landmark_observation_oracle(self, info_dict):
        assert self.route_info is not None
        # calculate the distance between the current position and the destination
        dest = self.route_info['dest_place']
        dist_to_dest = geocode_utils.calculate_distance_from_geocode(self.current_geocode, dest['geocode'])
        if dist_to_dest < self.cfg.LANDMARK_DETECT.ORACLE_RADIUS:
            info_dict['dest'] = True
        else:
            info_dict['dest'] = False
        
        # search nearby landmark for intersection
        keypoint_list = self.route_info['route_results']['geocode_list'][:-1]
        if len(keypoint_list) > 0:
            dist_keypoint_list = geocode_utils.cal_distance_between_two_position_list([self.current_geocode], keypoint_list)[0]
            min_idx_keypoint = np.argmin(dist_keypoint_list)
            min_dist_keypoint = dist_keypoint_list[min_idx_keypoint]
            
            if min_dist_keypoint > self.cfg.LANDMARK_DETECT.KEYPOINT_RADIUS and not info_dict['intersection'] and \
                    not info_dict['dest']:
                return "No landmarks nearby"
        
        dist_landmark_list = []
        for landmark in self.landmark_list:
            if landmark is not None:
                distance = geocode_utils.calculate_distance_from_geocode(self.current_geocode, landmark['geocode'])
                dist_landmark_list.append(distance)
            else:
                dist_landmark_list.append(1000000)

        min_idx = np.argmin(dist_landmark_list)
        min_distance = dist_landmark_list[min_idx]

        if min_distance < self.cfg.LANDMARK_DETECT.ORACLE_RADIUS:
            landmark = self.landmark_list[min_idx]
        else:
            return "No landmarks nearby"

        # calculate the spatial relationship
        heading = geocode_utils.calculate_heading_between_geocodes(self.current_geocode, landmark['geocode'])
        spatial = geocode_utils.calculate_spatial_relationship_with_headings(self.current_heading, heading)
        return f"{landmark['name']} is on your {spatial}"


    def move_on_rail_legacy(self, info_dict):
        action = info_dict['action']
        heading_list = info_dict['heading_list']
        if action in ['forward()']:
            move_idx = geocode_utils.select_argmin_heading_from_heading_list(self.target_heading, heading_list)
            heading_diff = geocode_utils.cal_min_heading_diff_between_headings(self.target_heading, heading_list[move_idx])
            if heading_diff > self.cfg.get('MAX_HEADING_DIFF', 45):
                heading_list = self.platform.mover.get_all_suitable_heading_to_path_vln(
                    self.current_geocode, radius_query=True
                )
                move_idx = geocode_utils.select_argmin_heading_from_heading_list(self.target_heading, heading_list)

            # self.platform.mover.adjust_heading_web(heading_list[move_idx])
            self.current_geocode = self.platform.mover.move(move_idx, old_geocode = self.current_geocode)
            self.current_heading = heading_list[move_idx]
        elif 'turn_direction' in action:
            direction = action.split('(')[1].split(')')[0]
            if direction in geocode_utils.DIRECTION_SET_ABS:
                # to change the agent heading
                orientation = action.split('(')[1].split(')')[0]
                orientation_idx = self.orientation_set.index(orientation)
                self.current_heading = self.orientation_heading[orientation_idx]
                self.target_heading = self.current_heading

                # self.platform.mover.adjust_heading_web(self.current_heading)
                
                # forward
                heading_list = self.platform.mover.get_all_suitable_heading_to_path_vln(
                    self.current_geocode, radius_query=True
                )
                move_idx = geocode_utils.select_argmin_heading_from_heading_list(self.target_heading, heading_list)
                # self.platform.mover.adjust_heading_web(heading_list[move_idx])
                self.current_geocode = self.platform.mover.move(move_idx, old_geocode = self.current_geocode)
                self.current_heading = heading_list[move_idx]
        else:
            # to handle 'stop()' or error action not in the action list
            pass

        print(f'>>> VisionLanguageNavigator: after moving, current geocode is: {self.current_geocode}')
        
    def move(self, info_dict):
        action = info_dict['action']
        heading_list = info_dict['heading_list']
        if action in ['forward()']:
            move_idx = geocode_utils.select_argmin_heading_from_heading_list(self.target_heading, heading_list)
            heading_diff = geocode_utils.cal_min_heading_diff_between_headings(self.target_heading, heading_list[move_idx])
            if heading_diff > self.cfg.get('MAX_HEADING_DIFF', 45):
                heading_list = self.platform.mover.get_all_suitable_heading_to_path_vln(
                    self.current_geocode, radius_query=True
                )
                move_idx = geocode_utils.select_argmin_heading_from_heading_list(self.target_heading, heading_list)

            # self.platform.mover.adjust_heading_web(heading_list[move_idx])
            self.current_geocode = self.platform.mover.move(move_idx, old_geocode = self.current_geocode)
            self.current_heading = heading_list[move_idx]
        elif 'turn_direction' in action:
            direction = action.split('(')[1].split(')')[0]
            if direction in geocode_utils.DIRECTION_SET_ABS:
                # to change the agent heading
                orientation = action.split('(')[1].split(')')[0]
                orientation_idx = self.orientation_set.index(orientation)
                self.current_heading = self.orientation_heading[orientation_idx]
                self.target_heading = self.current_heading

                # self.platform.mover.adjust_heading_web(self.current_heading)
                
                # forward
                heading_list = self.platform.mover.get_all_suitable_heading_to_path_vln(
                    self.current_geocode, radius_query=True
                )
                move_idx = geocode_utils.select_argmin_heading_from_heading_list(self.target_heading, heading_list)
                # self.platform.mover.adjust_heading_web(heading_list[move_idx])
                self.current_geocode = self.platform.mover.move(move_idx, old_geocode = self.current_geocode)
                self.current_heading = heading_list[move_idx]
        else:
            # to handle 'stop()' or error action not in the action list
            pass

        print(f'>>> VisionLanguageNavigator: after moving, current geocode is: {self.current_geocode}')
    


