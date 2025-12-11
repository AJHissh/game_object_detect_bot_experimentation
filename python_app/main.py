import cv2
import numpy as np
import time
import threading
import pyautogui
import keyboard
import math
import ctypes
import winsound
import os
import random
from cpp_bridge import CppController
from yolo_detector import YOLODetector

keep_running = True
is_paused = False

class FastObjectTracker:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.next_id = 1
        self.tracked_objects = {}
        self.max_frames_missing = 40
        
    def update(self, detected_objects):
        current_frame_objects = {}
        used_ids = set()
        
        for obj in detected_objects:
            screen_x = self.bot.DETECTION_LEFT + obj['center'][0]
            screen_y = self.bot.DETECTION_TOP + obj['center'][1]
            obj['screen_center'] = (screen_x, screen_y)
            
            dx = screen_x - self.bot.center_x
            dy = screen_y - self.bot.center_y
            obj['distance_from_player'] = math.sqrt(dx*dx + dy*dy)
            
            best_match_id = None
            best_distance = float('inf')
            
            for obj_id, tracked_obj in self.tracked_objects.items():
                if obj_id in used_ids:
                    continue
                
                tracked_pos = tracked_obj['screen_center']
                distance = math.sqrt((screen_x - tracked_pos[0])**2 + (screen_y - tracked_pos[1])**2)
                
                if (obj['class_name'] == tracked_obj['class_name'] and 
                    distance < 80 and distance < best_distance): 
                    best_match_id = obj_id
                    best_distance = distance
            
            if best_match_id is not None:
                obj['id'] = best_match_id
                self.tracked_objects[best_match_id] = obj
                self.tracked_objects[best_match_id]['frames_missing'] = 0
                used_ids.add(best_match_id)
                current_frame_objects[best_match_id] = obj
            else:
                obj_id = self.next_id
                self.next_id += 1
                obj['id'] = obj_id
                obj['frames_missing'] = 0
                self.tracked_objects[obj_id] = obj
                current_frame_objects[obj_id] = obj
        
        objects_to_remove = []
        for obj_id, tracked_obj in self.tracked_objects.items():
            if obj_id not in used_ids:
                tracked_obj['frames_missing'] += 1
                if tracked_obj['frames_missing'] > self.max_frames_missing:
                    objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
        
        return current_frame_objects

class OptimizedMonsterBot:
    def __init__(self, model_path):
        self.cpp = CppController()
        self.detector = YOLODetector(model_path)
        self.object_tracker = FastObjectTracker(self)
        
        self.screen_width, self.screen_height = self.cpp.get_screen_resolution()
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2
        
        self.DETECTION_WIDTH = 1400
        self.DETECTION_HEIGHT = 1000
        self.DETECTION_LEFT = self.center_x - (self.DETECTION_WIDTH // 2)
        self.DETECTION_TOP = self.center_y - (self.DETECTION_HEIGHT // 2)
        
        self.CAPTURE_FPS = 120
        self.PROCESSING_FPS = 120
        
        self.search_click_delay = 0.05 
        self.target_click_delay = 0.10  
        
        self.class_confidence_thresholds = {
            'monster': 0.35,
            'Monster': 0.35,
            'farm': 0.20, 
            'Farm': 0.20, 
            'Portal': 0.50,
            'portal': 0.50,
            'human': 0.25,
            'Human': 0.25,
        }
        
        self.default_confidence_threshold = 0.50
        self.PAUSE_KEY = 'f2'
        
        self.debug_monitor_enabled = False
        self.debug_window_name = "Detection Debug Monitor"
        self.debug_scale = 0.6
        self.last_debug_update = 0
        self.debug_update_interval = 0.1
        
  
        self.target_lock_proximity = 300
        self.locked_target_id = None
        self.locked_target = None
        self.locked_target_position = None
        self.locked_target_class = None
        self.target_lock_frames = 0
        self.max_target_lock_frames = 30 
        self.last_lock_lost_time = 0
        
 
        self.post_loot_check_frames = 0
        self.max_post_loot_check_frames = 30 
        self.last_known_monster_position = None
        self.monster_disappeared_time = 0
        self.loot_search_radius = 300 
        self.original_monster_position = None 
        
   
        self.post_lock_wait_frames = 0
        self.max_post_lock_wait_frames = 25 
        self.post_lock_wait_radius = 300 
        
   
        self.close_target_radius = 300
        self.close_target_constant_click = True
        self.last_close_target_click_time = 0
        self.close_target_click_interval = 0.05
        
   
        self.close_proximity_stability_threshold = 1  
        self.min_lock_frames_for_stability = 1  
        self.current_lock_frames = 0 
        

        self.last_player_position = None
        self.last_player_movement_time = time.time()
        self.player_stuck_threshold = 10.0  
        self.player_position_tolerance = 5 
        self.center_player_tracking_radius = 85  
        

        self.last_f9_press_time = time.time()
        self.f9_press_interval = 180.0  
        
        self.active_timers = {}
        
        self.search_direction = random.randint(0, 3)
        self.clicks_in_current_direction = 0
        self.min_clicks_per_direction = 15 
        self.max_clicks_per_direction = 30 
        self.consecutive_no_targets = 0
        self.search_trigger_threshold = 15 
        self.last_search_click_time = 0
        self.search_click_cooldown = 0.05  
        
        self.max_search_distance = 250 
        self.min_search_distance = 195  
        
        self.dynamic_loot_search_enabled = True
        self.last_player_center_position = (self.center_x, self.center_y)
        self.loot_position_offset = (0, 0)  
        
        self.portal_avoidance_enabled = True
        self.last_portal_detection_time = 0
        self.portal_avoidance_cooldown = 1.5 
        self.portal_avoidance_in_progress = False
        self.portal_avoidance_clicks_remaining = 0
        self.portal_avoidance_direction = None
        
        self.black_tile_detection_enabled = False
        self.black_tile_threshold = 50  
        self.black_tile_area_threshold = 50 
        self.last_black_tile_detection_time = 0
        self.black_tile_cooldown = 1.0  
        self.black_tile_avoidance_count = 0
        self.max_black_tile_avoidances = 3  
        
        self.player_bbox_size = 10  

        self.target_movement_history = {} 
        self.max_movement_history = 3 
        self.movement_prediction_frames = 3 
        
        self.last_f8_press_time = time.time()
        self.f8_press_interval = 1800.0 
    
        self.last_f7_press_time = time.time()
        self.f7_press_interval = 180.0   
        
        self.detection_stats = {
            'total_frames': 0,
            'detection_frames': 0,
            'current_fps': 0,
            'last_processing_time': 0,
            'locked_target_frames': 0,
            'close_target_clicks': 0,
            'farm_objects_targeted': 0,
            'search_moves': 0,
            'close_targets_targeted': 0,
            'far_targets_targeted': 0,
            'lock_recoveries': 0,
            'post_loot_checks': 0,
            'monster_locks': 0,
            'player_stuck_resets': 0,
            'loot_found': 0,
            'search_distance_violations': 0,
            'dynamic_loot_adjustments': 0,
            'portal_avoidances': 0,
            'black_tile_avoidances': 0,
            'post_lock_wait_targets': 0,
            'portal_emergency_escapes': 0,
            'black_tile_forced_directions': 0,
            'player_bbox_adjustments': 0,
            'movement_predictions': 0,
            'predicted_targets_hit': 0,
            'stable_locks_maintained': 0,
            'unnecessary_switches_prevented': 0,
            'auto_f9_presses': 0,  
            'auto_f8_presses': 0, 
            'auto_f7_presses': 0, 
        }
        
        print("=== ENHANCED BOT ===")
        print("   1. FARM OBJECTS (HIGHEST PRIORITY)")
        print("   2. CLOSE PROXIMITY Objects (with stability threshold)")
        print("   3. Locked Objects")
        print("   4. Any Object")

    def auto_press_f7_f8(self):
        current_time = time.time()
        pressed_any = False

        if current_time - self.last_f8_press_time >= self.f8_press_interval:
            print(f"⏰ 30-minute timer reached - pressing F8")
            pyautogui.press('f8')
            self.last_f8_press_time = current_time
            self.detection_stats['auto_f8_presses'] += 1
            pressed_any = True

        if current_time - self.last_f7_press_time >= self.f7_press_interval:
            print(f"⏰ 2-minute timer reached - pressing F7")
            pyautogui.press('f7')
            self.last_f7_press_time = current_time
            self.detection_stats['auto_f7_presses'] += 1
            pressed_any = True

        return pressed_any

    def auto_press_f9(self):
        current_time = time.time()

        if current_time - self.last_f9_press_time >= self.f9_press_interval:
            print(f"⏰ 3-minute timer reached - pressing F4")
            pyautogui.press('f4') 
            self.last_f9_press_time = current_time
            self.detection_stats['auto_f9_presses'] += 1
            return True

        if (self.last_player_position and 
            current_time - self.last_player_movement_time >= self.player_stuck_threshold):
            print(f"🚨 PLAYER STUCK for {self.player_stuck_threshold} seconds - pressing F9")
            pyautogui.press('f9') 
            self.last_player_movement_time = current_time
            self.last_f9_press_time = current_time
            self.detection_stats['player_stuck_resets'] += 1
            self.detection_stats['auto_f9_presses'] += 1
            threading.Timer(1.0, lambda: pyautogui.hotkey('alt', 'end')).start()
            return True

        return False

    def adjust_player_bounding_box(self, detected_objects):
        """Make player/human bounding boxes very small (20x20 pixels)"""
        adjusted_objects = []
        
        for obj in detected_objects:
            if self.is_human_class(obj):
                adjusted_obj = obj.copy()
                
                x1, y1, x2, y2 = obj['bbox']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                new_x1 = center_x - self.player_bbox_size / 2
                new_y1 = center_y - self.player_bbox_size / 2
                new_x2 = center_x + self.player_bbox_size / 2
                new_y2 = center_y + self.player_bbox_size / 2
                
                adjusted_obj['bbox'] = [new_x1, new_y1, new_x2, new_y2]
                adjusted_obj['center'] = [center_x, center_y]

                adjusted_obj['bbox_adjusted'] = True
                
                adjusted_objects.append(adjusted_obj)
                self.detection_stats['player_bbox_adjustments'] += 1
            else:
                adjusted_objects.append(obj)
        
        return adjusted_objects

    def update_target_movement_history(self, tracked_objects):
        current_time = time.time()
        
        for obj_id, obj in tracked_objects.items():
            if obj_id not in self.target_movement_history:
                self.target_movement_history[obj_id] = []
            
            current_pos = obj['screen_center']
            self.target_movement_history[obj_id].append({
                'position': current_pos,
                'timestamp': current_time,
                'velocity': (0, 0)  
            })
            
            if len(self.target_movement_history[obj_id]) > 1:
                prev_pos = self.target_movement_history[obj_id][-2]['position']
                time_diff = current_time - self.target_movement_history[obj_id][-2]['timestamp']
                
                if time_diff > 0:
                    dx = current_pos[0] - prev_pos[0]
                    dy = current_pos[1] - prev_pos[1]
                    velocity = (dx / time_diff, dy / time_diff)
                    self.target_movement_history[obj_id][-1]['velocity'] = velocity

            if len(self.target_movement_history[obj_id]) > self.max_movement_history:
                self.target_movement_history[obj_id] = self.target_movement_history[obj_id][-self.max_movement_history:]

        objects_to_remove = []
        for obj_id in self.target_movement_history:
            if obj_id not in tracked_objects:
                objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.target_movement_history[obj_id]

    def predict_target_position(self, obj_id, frames_ahead=3):
        """Predict future position of a target based on movement history"""
        if obj_id not in self.target_movement_history:
            return None
        
        history = self.target_movement_history[obj_id]
        if len(history) < 2:
            return None

        current_pos = history[-1]['position']
        current_vel = history[-1]['velocity']

        predicted_x = current_pos[0] + current_vel[0] * frames_ahead * (1.0 / self.PROCESSING_FPS)
        predicted_y = current_pos[1] + current_vel[1] * frames_ahead * (1.0 / self.PROCESSING_FPS)

        predicted_x = max(50, min(self.screen_width - 50, predicted_x))
        predicted_y = max(50, min(self.screen_height - 50, predicted_y))
        
        self.detection_stats['movement_predictions'] += 1
        return (predicted_x, predicted_y)

    def get_smooth_search_position(self):
        base_distance = (self.min_search_distance + self.max_search_distance) // 2
        
        distance_variation = random.randint(-20, 20)
        distance = base_distance + distance_variation
        
        angle_offset = (self.clicks_in_current_direction / self.max_clicks_per_direction) * 30  
        
        if self.search_direction == 0:  
            base_angle = 0
            search_x = self.center_x + distance
            search_y = self.center_y + int(math.sin(math.radians(base_angle + angle_offset)) * 150)
        elif self.search_direction == 1:  
            base_angle = 90
            search_x = self.center_x + int(math.cos(math.radians(base_angle + angle_offset)) * 150)
            search_y = self.center_y + distance
        elif self.search_direction == 2:  
            base_angle = 180
            search_x = self.center_x - distance
            search_y = self.center_y + int(math.sin(math.radians(base_angle + angle_offset)) * 150)
        else: 
            base_angle = 270
            search_x = self.center_x + int(math.cos(math.radians(base_angle + angle_offset)) * 150)
            search_y = self.center_y - distance
        
        search_x = max(50, min(self.screen_width - 50, search_x))
        search_y = max(50, min(self.screen_height - 50, search_y))
        
        return search_x, search_y

    def detect_black_tile(self, target_x, target_y):
        """Enhanced black tile detection - prevents infinite loops"""
        if not self.black_tile_detection_enabled:
            return False
            
        current_time = time.time()
        if current_time - self.last_black_tile_detection_time < self.black_tile_cooldown:
            return False
            
        try:
            capture_size = 50
            capture_left = max(0, int(target_x - capture_size // 2))
            capture_top = max(0, int(target_y - capture_size // 2))
            capture_width = min(capture_size, self.screen_width - capture_left)
            capture_height = min(capture_size, self.screen_height - capture_top)
            
            if capture_width <= 0 or capture_height <= 0:
                return False
                
            screenshot = pyautogui.screenshot(region=(
                capture_left, capture_top, capture_width, capture_height
            ))
            screen_frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            gray_frame = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2GRAY)
            
            dark_pixels = gray_frame < self.black_tile_threshold
            dark_area = np.sum(dark_pixels)

            if dark_area > self.black_tile_area_threshold:
                self.last_black_tile_detection_time = current_time
                self.detection_stats['black_tile_avoidances'] += 1
                self.black_tile_avoidance_count += 1
                print(f"⬛ BLACK TILE DETECTED! Area: {dark_area} pixels - Avoiding position")
                return True
                
        except Exception as e:
            print(f"Black tile detection error: {e}")
            
        return False

    def handle_black_tile_avoidance(self, old_direction):
        if self.black_tile_avoidance_count >= self.max_black_tile_avoidances:
            print(f"🔄 Too many black tile avoidances ({self.black_tile_avoidance_count}), forcing RIGHT direction")
            self.search_direction = 0  
            self.black_tile_avoidance_count = 0
            self.clicks_in_current_direction = 0
            self.max_clicks_per_direction = random.randint(self.min_clicks_per_direction, self.max_clicks_per_direction)
            self.detection_stats['black_tile_forced_directions'] += 1
            return
        
        possible_directions = [0, 1, 2, 3]
        opposite_direction = (old_direction + 2) % 4

        possible_directions.remove(opposite_direction)

        self.search_direction = random.choice(possible_directions)
        self.clicks_in_current_direction = 0
        self.max_clicks_per_direction = random.randint(self.min_clicks_per_direction, self.max_clicks_per_direction)
        
        direction_names = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP"}
        print(f"⬛ BLACK TILE AVOIDED! Changing direction from {direction_names[old_direction]} to {direction_names[self.search_direction]}")

    def update_dynamic_loot_position(self, current_player_position):
        if not self.dynamic_loot_search_enabled:
            return
            
        self.last_player_center_position = current_player_position

    def get_current_loot_search_position(self):
        return (self.center_x, self.center_y)

    def avoid_portal(self, tracked_objects):
        """Enhanced portal avoidance - PAUSE and move 5 clicks away immediately"""
        if not self.portal_avoidance_enabled:
            return False           
        current_time = time.time()
        if self.portal_avoidance_in_progress:
            if self.portal_avoidance_clicks_remaining > 0:
                if self.perform_portal_escape_click():
                    self.portal_avoidance_clicks_remaining -= 1
                    print(f"🚫 PORTAL ESCAPE: {self.portal_avoidance_clicks_remaining} clicks remaining")
                return True
            else:
                self.portal_avoidance_in_progress = False
                self.portal_avoidance_clicks_remaining = 0
                self.portal_avoidance_direction = None
                self.last_portal_detection_time = current_time
                print("✅ Portal escape complete - resuming normal operation")
                return False
        
        if current_time - self.last_portal_detection_time < self.portal_avoidance_cooldown:
            return False

        portals = []
        for obj_id, obj in tracked_objects.items():
            if obj['class_name'].lower() == 'portal':
                portals.append(obj)
        
        if not portals:
            return False

        closest_portal = min(portals, key=lambda x: x['distance_from_player'])
        portal_distance = closest_portal['distance_from_player']
        
        print(f"🚨 PORTAL DETECTED! Distance: {portal_distance:.0f}px - INITIATING EMERGENCY ESCAPE")

        self.portal_avoidance_in_progress = True
        self.portal_avoidance_clicks_remaining = 20  
        self.detection_stats['portal_emergency_escapes'] += 1

        portal_x, portal_y = closest_portal['screen_center']
        dx = portal_x - self.center_x
        dy = portal_y - self.center_y
        
        if abs(dx) > abs(dy):
            if dx > 0:
                self.portal_avoidance_direction = 2 
            else:
                self.portal_avoidance_direction = 0 
        else:
            if dy > 0:
                self.portal_avoidance_direction = 3  
            else:
                self.portal_avoidance_direction = 1  
        
        direction_names = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP"}
        print(f"🚫 Moving {direction_names[self.portal_avoidance_direction]} away from portal (5 clicks)")
        
        return True

    def perform_portal_escape_click(self):
        if self.portal_avoidance_direction is None:
            return False
        escape_distance = 150 
        direction = self.portal_avoidance_direction
        
        if direction == 0:  
            escape_x = self.center_x + escape_distance
            escape_y = self.center_y + random.randint(-50, 50)
        elif direction == 1: 
            escape_x = self.center_x + random.randint(-50, 50)
            escape_y = self.center_y + escape_distance
        elif direction == 2: 
            escape_x = self.center_x - escape_distance
            escape_y = self.center_y + random.randint(-50, 50)
        else: 
            escape_x = self.center_x + random.randint(-50, 50)
            escape_y = self.center_y - escape_distance

        escape_x = max(50, min(self.screen_width - 50, escape_x))
        escape_y = max(50, min(self.screen_height - 50, escape_y))

        if self.cpp.move_mouse(escape_x, escape_y):
            time.sleep(0.1) 
            if self.cpp.click():
                return True
        
        return False

    def get_confidence_threshold_for_class(self, class_name):
        if class_name in self.class_confidence_thresholds:
            return self.class_confidence_thresholds[class_name]
        
        class_name_lower = class_name.lower()
        for known_class, threshold in self.class_confidence_thresholds.items():
            if known_class.lower() == class_name_lower:
                return threshold
        
        return self.default_confidence_threshold

    def is_monster_class(self, obj):
        return obj['class_name'].lower() in ['monster', 'enemy', 'mob']

    def is_human_class(self, obj):
        return obj['class_name'].lower() in ['human', 'player', 'character']

    def is_farm_class(self, obj):
        return obj['class_name'].lower() in ['farm', 'crop', 'resource', 'loot']

    def detect_player_movement(self, tracked_objects):
        current_time = time.time()

        center_player = None
        for obj_id, obj in tracked_objects.items():
            if self.is_human_class(obj):
                dx = obj['screen_center'][0] - self.center_x
                dy = obj['screen_center'][1] - self.center_y
                distance_from_center = math.sqrt(dx*dx + dy*dy)
                
                if distance_from_center < self.center_player_tracking_radius: 
                    center_player = obj
                    break
        
        if center_player:
            current_pos = center_player['screen_center']
            
            if self.last_player_position is None:
                self.last_player_position = current_pos
                self.last_player_movement_time = current_time
                return True

            dx = current_pos[0] - self.last_player_position[0]
            dy = current_pos[1] - self.last_player_position[1]
            movement_distance = math.sqrt(dx*dx + dy*dy)
            
            if movement_distance > self.player_position_tolerance:
                self.last_player_position = current_pos
                self.last_player_movement_time = current_time
                if self.dynamic_loot_search_enabled and self.last_known_monster_position:
                    self.update_dynamic_loot_position(current_pos)
                
                return True
            else:
                time_since_movement = current_time - self.last_player_movement_time
                if time_since_movement > self.player_stuck_threshold:
                    pass
        else:
            self.last_player_position = None
        
        return False

    def find_best_target(self, tracked_objects):
        if not tracked_objects:
            if self.locked_target_id:
                print(f"🔓 Clearing stuck lock - no objects detected")
                self.locked_target_id = None
                self.locked_target = None
                self.locked_target_position = None
                self.locked_target_class = None
                self.target_lock_frames = 0
                self.current_lock_frames = 0
            return None
            
        current_target_still_exists = False
        if self.locked_target_id and self.locked_target_id in tracked_objects:
            current_target_still_exists = True
            self.target_lock_frames = 0

        farm_objects = {}
        for obj_id, obj in tracked_objects.items():
            if self.is_human_class(obj) or obj['class_name'].lower() == 'portal':
                continue

            if not self.is_farm_class(obj):
                continue

            class_threshold = self.get_confidence_threshold_for_class(obj['class_name'])
            if obj['confidence'] < class_threshold:
                continue

            if self.locked_target_position:
                dx = obj['screen_center'][0] - self.locked_target_position[0]
                dy = obj['screen_center'][1] - self.locked_target_position[1]
                distance_from_focus = math.sqrt(dx*dx + dy*dy)
                if distance_from_focus > 300:
                    continue

            farm_objects[obj_id] = obj
            
        if farm_objects and (not self.locked_target_id or not current_target_still_exists):
            farm_list = list(farm_objects.values())
            farm_list.sort(key=lambda x: x['distance_from_player'])
            closest_farm = farm_list[0]

            should_switch_to_farm = (
                not self.locked_target_id or 
                not current_target_still_exists or
                closest_farm['distance_from_player'] < 150 
            )

            if should_switch_to_farm:
                print(f"🚜 FARM PRIORITY: Switching to farm object ID:{closest_farm['id']} "
                      f"({closest_farm['distance_from_player']:.0f}px)")
                self.locked_target_id = closest_farm['id']
                self.locked_target = closest_farm
                self.locked_target_position = closest_farm['screen_center']
                self.locked_target_class = closest_farm['class_name']
                self.target_lock_frames = 0
                self.post_loot_check_frames = 0
                self.current_lock_frames = 0
                self.detection_stats['farm_objects_targeted'] += 1
                return closest_farm

        if self.locked_target_id and current_target_still_exists:
            locked_obj = tracked_objects[self.locked_target_id]
            self.locked_target = locked_obj
            self.locked_target_position = locked_obj['screen_center']
            self.locked_target_class = locked_obj['class_name']
            self.target_lock_frames = 0
            self.post_loot_check_frames = 0
            self.current_lock_frames += 1
            self.detection_stats['stable_locks_maintained'] += 1

            if self.target_lock_frames > 0:
                print(f"🎯 LOCK RECOVERED: ID:{self.locked_target_id} was temporarily lost")

            return locked_obj

        close_objects = {}
        for obj_id, obj in tracked_objects.items():
            if self.is_human_class(obj) or obj['class_name'].lower() == 'portal':
                continue

            if not (obj['is_monster'] or obj['is_farm']):
                continue

            class_threshold = self.get_confidence_threshold_for_class(obj['class_name'])
            if obj['confidence'] < class_threshold:
                continue

            if obj['distance_from_player'] <= self.close_target_radius:
                close_objects[obj_id] = obj

        if close_objects:
            close_list = list(close_objects.values())

            if self.locked_target_position:
                for obj in close_list:
                    dx = obj['screen_center'][0] - self.locked_target_position[0]
                    dy = obj['screen_center'][1] - self.locked_target_position[1]
                    obj['distance_from_focus'] = math.sqrt(dx*dx + dy*dy)
                close_list.sort(key=lambda x: x['distance_from_focus'])
            else:
                close_list.sort(key=lambda x: x['distance_from_player'])

            closest_target = close_list[0]
            should_switch = False
            if self.locked_target_position:
                if closest_target.get('distance_from_focus', 0) < 200:
                    should_switch = True
                elif closest_target['distance_from_player'] < (self.locked_target['distance_from_player'] - 100):
                    should_switch = True
            else:
                should_switch = True

            if should_switch and self.current_lock_frames >= self.min_lock_frames_for_stability:
                current_target_distance = float('inf')
                if self.locked_target_id and self.locked_target_id in tracked_objects:
                    current_target_distance = tracked_objects[self.locked_target_id]['distance_from_player']

                new_target_distance = closest_target['distance_from_player']
                distance_improvement = current_target_distance - new_target_distance

                area_bonus = 50 if closest_target.get('distance_from_focus', 0) < 200 else 0
                effective_improvement = distance_improvement + area_bonus

                if effective_improvement >= self.close_proximity_stability_threshold:
                    print(f"🎯 ENHANCED CLOSE PROXIMITY: Switching to closer target ID:{closest_target['id']} "
                          f"({new_target_distance:.0f}px, {distance_improvement:.0f}px improvement)")
                    self.locked_target_id = closest_target['id']
                    self.locked_target = closest_target
                    self.locked_target_position = closest_target['screen_center']
                    self.locked_target_class = closest_target['class_name']
                    self.target_lock_frames = 0
                    self.post_loot_check_frames = 0
                    self.current_lock_frames = 0
                    self.detection_stats['close_targets_targeted'] += 1
                    return closest_target
                else:
                    self.detection_stats['unnecessary_switches_prevented'] += 1
                    if distance_improvement > 0:
                        print(f"⚖️ AREA STABILITY: Maintaining focus area "
                              f"(improvement: {distance_improvement:.0f}px)")

        if self.locked_target_id and not current_target_still_exists:
            self.target_lock_frames += 1

            max_forgiveness_frames = 8 if self.is_monster_class(self.locked_target) else 5

            if self.target_lock_frames <= max_forgiveness_frames:
                same_area_objects = {}
                for obj_id, obj in tracked_objects.items():
                    if self.is_human_class(obj) or obj['class_name'].lower() == 'portal':
                        continue

                    if not (obj['is_monster'] or obj['is_farm']):
                        continue

                    class_threshold = self.get_confidence_threshold_for_class(obj['class_name'])
                    if obj['confidence'] < class_threshold:
                        continue
                    
                    if self.locked_target_position:
                        dx = obj['screen_center'][0] - self.locked_target_position[0]
                        dy = obj['screen_center'][1] - self.locked_target_position[1]
                        distance_from_lost_target = math.sqrt(dx*dx + dy*dy)

                        if distance_from_lost_target < 250: 
                            same_area_objects[obj_id] = obj

                if same_area_objects:
                    area_list = list(same_area_objects.values())
                    area_list.sort(key=lambda x: x['distance_from_player'])
                    best_area_target = area_list[0]

                    print(f"🎯 AREA-BASED SWITCH: Lost ID:{self.locked_target_id}, switching to nearby ID:{best_area_target['id']}")
                    self.locked_target_id = best_area_target['id']
                    self.locked_target = best_area_target
                    self.locked_target_position = best_area_target['screen_center']
                    self.locked_target_class = best_area_target['class_name']
                    self.target_lock_frames = 0
                    self.current_lock_frames = 0
                    return best_area_target
                else:
                    if self.target_lock_frames % 3 == 0:
                        print(f"🔍 Waiting for target ID:{self.locked_target_id} to reappear... "
                              f"({self.target_lock_frames}/{max_forgiveness_frames})")
                    return None
            else:
                print(f"🔓 Lock lost on ID:{self.locked_target_id} after {self.target_lock_frames} frames")

                if self.is_monster_class(self.locked_target):
                    self.last_known_monster_position = (self.center_x, self.center_y)
                    self.original_monster_position = (self.center_x, self.center_y)
                    self.monster_disappeared_time = time.time()
                    self.post_loot_check_frames = 0
                    self.last_player_center_position = (self.center_x, self.center_y)
                    print(f"💰 Monster disappeared - starting STATIC loot detection around player")

                self.post_lock_wait_frames = 0
                self.last_lock_lost_time = time.time()
                self.locked_target_id = None
                self.locked_target = None
                self.locked_target_position = None
                self.locked_target_class = None
                self.current_lock_frames = 0

        if self.last_lock_lost_time and self.post_lock_wait_frames < self.max_post_lock_wait_frames:
            self.post_lock_wait_frames += 1
            
            close_to_player_objects = {}
            for obj_id, obj in tracked_objects.items():
                if self.is_human_class(obj) or obj['class_name'].lower() == 'portal':
                    continue 
                    
                if not (obj['is_monster'] or obj['is_farm']):
                    continue
                    
                class_threshold = self.get_confidence_threshold_for_class(obj['class_name'])
                if obj['confidence'] < class_threshold:
                    continue

                if obj['distance_from_player'] <= self.post_lock_wait_radius:
                    close_to_player_objects[obj_id] = obj
            
            if close_to_player_objects:
                close_list = list(close_to_player_objects.values())
                close_list.sort(key=lambda x: x['distance_from_player'])
                best_target = close_list[0]
                
                self.locked_target_id = best_target['id']
                self.locked_target = best_target
                self.locked_target_position = best_target['screen_center']
                self.locked_target_class = best_target['class_name']
                self.target_lock_frames = 0
                self.post_lock_wait_frames = 0
                self.current_lock_frames = 0
                self.detection_stats['post_lock_wait_targets'] += 1
                
                target_type = "CLOSE" if best_target['distance_from_player'] <= self.close_target_radius else "FARM" if best_target['is_farm'] else "NEAR"
                print(f"🎯 POST-LOCK TARGET: {target_type} ID:{best_target['id']} ({best_target['distance_from_player']:.0f}px from player)")
                return best_target
            else:
                if self.post_lock_wait_frames % 5 == 0:  
                    print(f"⏳ Post-lock wait: {self.post_lock_wait_frames}/{self.max_post_lock_wait_frames} - No objects within {self.post_lock_wait_radius}px")
                return None

        if self.last_known_monster_position and self.post_loot_check_frames < self.max_post_loot_check_frames:
            self.post_loot_check_frames += 1
            self.detection_stats['post_loot_checks'] += 1
            
            current_loot_position = self.get_current_loot_search_position()

            nearby_objects = {}
            for obj_id, obj in tracked_objects.items():
                if not (obj['is_monster'] or obj['is_farm']):
                    continue
                
                class_threshold = self.get_confidence_threshold_for_class(obj['class_name'])
                if obj['confidence'] < class_threshold:
                    continue

                obj_pos = obj['screen_center']
                distance_to_player = math.sqrt((obj_pos[0]-current_loot_position[0])**2 + 
                                             (obj_pos[1]-current_loot_position[1])**2)
                
                if distance_to_player < self.loot_search_radius:
                    nearby_objects[obj_id] = obj
            
            if nearby_objects:
                farm_objects = {k: v for k, v in nearby_objects.items() if self.is_farm_class(v)}
                monster_objects = {k: v for k, v in nearby_objects.items() if self.is_monster_class(v)}
                
                if farm_objects:
                    farm_list = list(farm_objects.values())
                    farm_list.sort(key=lambda x: x['distance_from_player'])
                    best_target = farm_list[0]
                    self.detection_stats['loot_found'] += 1
                    print(f"💰 STATIC LOOT FOUND: Farm object {best_target['distance_from_player']:.0f}px from player")
                else:
                    monster_list = list(monster_objects.values())
                    monster_list.sort(key=lambda x: x['distance_from_player'])
                    best_target = monster_list[0] if monster_list else None
                
                if best_target:
                    self.locked_target_id = best_target['id']
                    self.locked_target = best_target
                    self.locked_target_position = best_target['screen_center']
                    self.locked_target_class = best_target['class_name']
                    self.target_lock_frames = 0
                    self.post_loot_check_frames = 0
                    self.current_lock_frames = 0 
                    print(f"💰 STATIC POST-LOOT: Locked {best_target['class_name']} ID:{best_target['id']}")
                    return best_target
            
            if self.post_loot_check_frames < self.max_post_loot_check_frames:
                if self.post_loot_check_frames % 3 == 0:
                    print(f"💰 Static loot detection: {self.post_loot_check_frames}/{self.max_post_loot_check_frames} - Searching {self.loot_search_radius}px around player")
                return None
            else:
                print("⏹️ Static loot detection complete - no objects found")
                self.last_known_monster_position = None
                self.original_monster_position = None
                self.post_loot_check_frames = 0
                
        valid_objects = {}
        for obj_id, obj in tracked_objects.items():
            if self.is_human_class(obj) or obj['class_name'].lower() == 'portal':
                continue 
                
            if not (obj['is_monster'] or obj['is_farm']):
                continue
                
            class_threshold = self.get_confidence_threshold_for_class(obj['class_name'])
            if obj['confidence'] < class_threshold:
                continue
                
            valid_objects[obj_id] = obj

        if not valid_objects:
            return None

        other_list = list(valid_objects.values())
        other_list.sort(key=lambda x: x['distance_from_player'])
        if other_list:
            best_target = other_list[0]
            self.detection_stats['far_targets_targeted'] += 1
        else:
            return None

        self.locked_target_id = best_target['id']
        self.locked_target = best_target
        self.locked_target_position = best_target['screen_center']
        self.locked_target_class = best_target['class_name']
        self.target_lock_frames = 0
        self.post_loot_check_frames = 0
        self.current_lock_frames = 0 

        if self.is_monster_class(best_target):
            self.detection_stats['monster_locks'] += 1
        
        target_type = "FARM" if self.is_farm_class(best_target) else "FAR"
        print(f"🎯 LOCKED {target_type}: ID:{best_target['id']} ({best_target['distance_from_player']:.0f}px)")
        
        return best_target

    def click_target_immediately(self, target_x, target_y):
        if self.detect_black_tile(target_x, target_y):
            old_direction = self.search_direction
            self.handle_black_tile_avoidance(old_direction)
            return False
        
        if self.locked_target_id and self.locked_target_id in self.target_movement_history:
            predicted_pos = self.predict_target_position(self.locked_target_id, self.movement_prediction_frames)
            if predicted_pos:
                pred_x, pred_y = predicted_pos
                distance_to_predicted = math.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2)
                if distance_to_predicted < 100:  
                    target_x, target_y = pred_x, pred_y
                    self.detection_stats['predicted_targets_hit'] += 1
        
        if self.cpp.move_mouse(target_x, target_y):
            time.sleep(self.target_click_delay)
            if self.cpp.click():
                return True
        return False

    def click_close_target_continuously(self, target_x, target_y):
        current_time = time.time()    
        if current_time - self.last_close_target_click_time < self.close_target_click_interval:
            return False
        if self.detect_black_tile(target_x, target_y):
            old_direction = self.search_direction
            self.handle_black_tile_avoidance(old_direction)
            return False
        
        if self.cpp.move_mouse(target_x, target_y):
            if self.cpp.click():
                self.last_close_target_click_time = current_time
                self.detection_stats['close_target_clicks'] += 1
                return True
        return False

    def should_perform_search(self):
        if (self.locked_target_id or 
            (self.last_lock_lost_time and self.post_lock_wait_frames < self.max_post_lock_wait_frames) or
            (self.last_known_monster_position and self.post_loot_check_frames < self.max_post_loot_check_frames)):
            return False
        if self.consecutive_no_targets > 120 : 
            print("🚨 EMERGENCY: Forcing search after 50 frames with no targets")
            return True
            
        return True

    def perform_smooth_search(self):
        if not self.should_perform_search():
            return False
            
        current_time = time.time()

        if current_time - self.last_search_click_time < self.search_click_cooldown:
            return False
            
        self.clicks_in_current_direction += 1
        
        if self.clicks_in_current_direction >= self.max_clicks_per_direction:
            self.search_direction = random.randint(0, 3)
            self.clicks_in_current_direction = 0
            self.max_clicks_per_direction = random.randint(self.min_clicks_per_direction, self.max_clicks_per_direction)
            direction_names = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP"}
            print(f"🔄 Changing search direction to: {direction_names[self.search_direction]}")
  
        search_x, search_y = self.get_smooth_search_position()

        distance_from_center = math.sqrt((search_x - self.center_x)**2 + (search_y - self.center_y)**2)

        if distance_from_center < self.min_search_distance or distance_from_center > self.max_search_distance:
            print(f"📏 Search position out of bounds ({distance_from_center:.0f}px), regenerating...")
            self.detection_stats['search_distance_violations'] += 1
            search_x, search_y = self.get_smooth_search_position()
            distance_from_center = math.sqrt((search_x - self.center_x)**2 + (search_y - self.center_y)**2)

        if self.detect_black_tile(search_x, search_y):
            old_direction = self.search_direction
            self.handle_black_tile_avoidance(old_direction)
            return False
        
        if self.cpp.move_mouse(search_x, search_y):
            time.sleep(self.search_click_delay)
            
            if self.cpp.click():
                self.detection_stats['search_moves'] += 1
                self.last_search_click_time = current_time
                direction_names = {0: "RIGHT", 1: "DOWN", 2: "LEFT", 3: "UP"}
                print(f"🔍 {direction_names[self.search_direction]} - Click {self.clicks_in_current_direction}/{self.max_clicks_per_direction} ({distance_from_center:.0f}px)")
                return True
        
        return False

    def update_visual_timers(self):
        current_time = time.time()
        self.active_timers.clear()
        
        time_since_last_f8 = current_time - self.last_f8_press_time
        progress = min(time_since_last_f8 / self.f8_press_interval, 1.0)
        self.active_timers["Auto F8 Timer"] = progress

        time_since_last_f7 = current_time - self.last_f7_press_time
        progress = min(time_since_last_f7 / self.f7_press_interval, 1.0)
        self.active_timers["Auto F7 Timer"] = progress
        
        if self.last_lock_lost_time and self.post_lock_wait_frames < self.max_post_lock_wait_frames:
            progress = self.post_lock_wait_frames / self.max_post_lock_wait_frames
            self.active_timers["Post-Lock Wait"] = progress

        if self.last_known_monster_position and self.post_loot_check_frames < self.max_post_loot_check_frames:
            progress = self.post_loot_check_frames / self.max_post_loot_check_frames
            self.active_timers["Post-Loot Check"] = progress

        if self.locked_target_id and self.target_lock_frames > 0:
            progress = self.target_lock_frames / self.max_target_lock_frames
            self.active_timers["Lock Persistence"] = progress

        if current_time - self.last_close_target_click_time < self.close_target_click_interval:
            progress = (current_time - self.last_close_target_click_time) / self.close_target_click_interval
            self.active_timers["Click Cooldown"] = progress

        if self.last_player_position:
            time_since_movement = current_time - self.last_player_movement_time
            progress = min(time_since_movement / self.player_stuck_threshold, 1.0)
            self.active_timers["Player Stuck"] = progress

        if current_time - self.last_search_click_time < self.search_click_cooldown:
            progress = (current_time - self.last_search_click_time) / self.search_click_cooldown
            self.active_timers["Search Cooldown"] = progress

        if self.portal_avoidance_in_progress:
            progress = (5 - self.portal_avoidance_clicks_remaining) / 5
            self.active_timers["Portal Escape"] = progress

        time_since_last_f9 = current_time - self.last_f9_press_time
        progress = min(time_since_last_f9 / self.f9_press_interval, 1.0)
        self.active_timers["Auto F9 Timer"] = progress

    def create_debug_display(self, frame, tracked_objects, processing_time):
        if not self.debug_monitor_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_debug_update < self.debug_update_interval:
            return
            
        self.last_debug_update = current_time
        
        try:
            debug_frame = frame.copy()

            self.detection_stats['total_frames'] += 1
            if processing_time > 0:
                self.detection_stats['current_fps'] = 1.0 / processing_time
            self.detection_stats['last_processing_time'] = processing_time
            
            if tracked_objects:
                self.detection_stats['detection_frames'] += 1
            if self.locked_target_id:
                self.detection_stats['locked_target_frames'] += 1

            self.update_visual_timers()

            center_x = self.DETECTION_WIDTH // 2
            center_y = self.DETECTION_HEIGHT // 2
            cv2.circle(debug_frame, (center_x, center_y), self.close_target_radius, (0, 165, 255), 2)
            cv2.putText(debug_frame, "CLOSE RANGE", (center_x - 50, center_y - self.close_target_radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            cv2.circle(debug_frame, (center_x, center_y), self.post_lock_wait_radius, (255, 165, 0), 2)
            cv2.putText(debug_frame, "POST-LOCK WAIT", (center_x - 60, center_y - self.post_lock_wait_radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

            cv2.circle(debug_frame, (center_x, center_y), self.max_search_distance, (255, 0, 255), 2)
            cv2.putText(debug_frame, "SEARCH LIMIT", (center_x - 50, center_y - self.max_search_distance - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            if self.last_known_monster_position:
                loot_x = center_x
                loot_y = center_y
                cv2.circle(debug_frame, (loot_x, loot_y), self.loot_search_radius, (255, 255, 0), 2)
                cv2.putText(debug_frame, "STATIC LOOT", (loot_x - 50, loot_y - self.loot_search_radius - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.circle(debug_frame, (center_x, center_y), self.center_player_tracking_radius, (255, 255, 255), 2)
            cv2.putText(debug_frame, "PLAYER TRACKING", (center_x - 60, center_y - self.center_player_tracking_radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for obj_id, obj in tracked_objects.items():
                x1, y1, x2, y2 = obj['bbox']
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if obj_id == self.locked_target_id:
                    color = (0, 255, 255) 
                elif self.is_farm_class(obj):
                    color = (0, 255, 0)   
                elif obj['distance_from_player'] <= self.close_target_radius:
                    color = (0, 165, 255) 
                elif self.is_human_class(obj):
                    color = (128, 0, 128) 
                elif obj['class_name'].lower() == 'portal':
                    color = (255, 0, 255)  
                else:
                    color = (0, 0, 255)    
                
                cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                
                class_name = obj['class_name']
                confidence = obj['confidence']

                class_label = f"{class_name}"
                class_text_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                class_bg_x1 = int(x1) - 5
                class_bg_y1 = int(y1) - class_text_size[1] - 10
                class_bg_x2 = int(x1) + class_text_size[0] + 10
                class_bg_y2 = int(y1) - 5

                cv2.rectangle(debug_frame, (class_bg_x1, class_bg_y1), (class_bg_x2, class_bg_y2), color, -1)
                cv2.putText(debug_frame, class_label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                conf_label = f"CONF: {confidence:.3f}"
                conf_text_size = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                conf_bg_x1 = int(x1) - 5
                conf_bg_y1 = int(y1) - class_text_size[1] - conf_text_size[1] - 15
                conf_bg_x2 = int(x1) + conf_text_size[0] + 10
                conf_bg_y2 = int(y1) - class_text_size[1] - 10

                cv2.rectangle(debug_frame, (conf_bg_x1, conf_bg_y1), (conf_bg_x2, conf_bg_y2), color, -1)
                cv2.putText(debug_frame, conf_label, (int(x1), int(y1) - class_text_size[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                id_label = f"ID:{obj_id} ({obj['distance_from_player']:.0f}px)"
                id_text_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                id_bg_x1 = int(x1) - 5
                id_bg_y1 = int(y2) + 5
                id_bg_x2 = int(x1) + id_text_size[0] + 10
                id_bg_y2 = int(y2) + id_text_size[1] + 10

                cv2.rectangle(debug_frame, (id_bg_x1, id_bg_y1), (id_bg_x2, id_bg_y2), color, -1)
                cv2.putText(debug_frame, id_label, (int(x1), int(y2) + id_text_size[1] + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if obj_id == self.locked_target_id:
                    cv2.circle(debug_frame, (int(center_x), int(center_y)), 50, (0, 255, 255), 4)
                    cv2.putText(debug_frame, "LOCKED", (int(center_x) - 40, int(center_y) - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                if self.is_farm_class(obj) and obj_id != self.locked_target_id:
                    cv2.circle(debug_frame, (int(center_x), int(center_y)), 70, (0, 255, 0), 4)
                    cv2.putText(debug_frame, "FARM!", (int(center_x) - 25, int(center_y) - 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                elif obj['distance_from_player'] <= self.close_target_radius and obj_id != self.locked_target_id:
                    cv2.circle(debug_frame, (int(center_x), int(center_y)), 60, (0, 165, 255), 4)
                    cv2.putText(debug_frame, "CLOSE!", (int(center_x) - 30, int(center_y) - 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)

                if obj.get('bbox_adjusted', False):
                    cv2.circle(debug_frame, (int(center_x), int(center_y)), 80, (255, 0, 0), 4)
                    cv2.putText(debug_frame, "ADJUSTED", (int(center_x) - 40, int(center_y) - 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)

                if obj_id == self.locked_target_id and obj_id in self.target_movement_history:
                    predicted_pos = self.predict_target_position(obj_id, self.movement_prediction_frames)
                    if predicted_pos:
                        pred_x, pred_y = predicted_pos
                        debug_pred_x = pred_x - self.DETECTION_LEFT
                        debug_pred_y = pred_y - self.DETECTION_TOP
                        
                        if (0 <= debug_pred_x < self.DETECTION_WIDTH and 
                            0 <= debug_pred_y < self.DETECTION_HEIGHT):
                            cv2.circle(debug_frame, (int(debug_pred_x), int(debug_pred_y)), 30, (0, 255, 0), 3)
                            cv2.putText(debug_frame, "PREDICTED", (int(debug_pred_x) - 50, int(debug_pred_y) - 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            info_panel = np.zeros((600, 600, 3), dtype=np.uint8)

            info_lines = [
                f"FPS: {self.detection_stats['current_fps']:.1f}",
                f"Frame: {self.detection_stats['total_frames']}",
                f"Proc Time: {processing_time*1000:.1f}ms",
                f"Objects: {len(tracked_objects)}",
                f"Locked Target: {self.locked_target_id if self.locked_target_id else 'None'}",
                f"Stability Frames: {self.current_lock_frames}",
                "",
                "=== TARGET STATS ===",
                f"FARM Targets: {self.detection_stats['farm_objects_targeted']}",
                f"Close Targets: {self.detection_stats['close_targets_targeted']}",
                f"Far Targets: {self.detection_stats['far_targets_targeted']}",
                f"Monster Locks: {self.detection_stats['monster_locks']}",
                f"Close Clicks: {self.detection_stats['close_target_clicks']}",
                f"Search Moves: {self.detection_stats['search_moves']}",
                f"Lock Recoveries: {self.detection_stats['lock_recoveries']}",
                f"Post-Loot Checks: {self.detection_stats['post_loot_checks']}",
                f"Loot Found: {self.detection_stats['loot_found']}",
                f"Post-Lock Targets: {self.detection_stats['post_lock_wait_targets']}",
                f"Player BBox Adjustments: {self.detection_stats['player_bbox_adjustments']}",
                f"Movement Predictions: {self.detection_stats['movement_predictions']}",
                f"Predicted Targets Hit: {self.detection_stats['predicted_targets_hit']}",
                f"Stable Locks Maintained: {self.detection_stats['stable_locks_maintained']}",
                f"Unnecessary Switches Prevented: {self.detection_stats['unnecessary_switches_prevented']}",
                "",
                "=== PLAYER STATS ===",
                f"Stuck Resets: {self.detection_stats['player_stuck_resets']}",
                f"Auto F9 Presses: {self.detection_stats['auto_f9_presses']}",
                "",
                "=== SEARCH STATS ===",
                f"Distance Violations: {self.detection_stats['search_distance_violations']}",
                f"Dynamic Adjustments: {self.detection_stats['dynamic_loot_adjustments']}",
                f"Portal Avoidances: {self.detection_stats['portal_avoidances']}",
                f"Portal Escapes: {self.detection_stats['portal_emergency_escapes']}",
                f"Black Tile Avoidances: {self.detection_stats['black_tile_avoidances']}",
                f"Black Tile Forced Directions: {self.detection_stats['black_tile_forced_directions']}",
                f"Search Range: {self.min_search_distance}-{self.max_search_distance}px",
                f"Post-Lock Radius: {self.post_lock_wait_radius}px",
                f"Player Tracking Radius: {self.center_player_tracking_radius}px",
                f"Player BBox Size: {self.player_bbox_size}x{self.player_bbox_size}px",
                f"Movement Prediction Frames: {self.movement_prediction_frames}",
                f"Stability Threshold: {self.close_proximity_stability_threshold}px",
                f"Min Lock Frames: {self.min_lock_frames_for_stability}",
                "",
                f"Search: {['RIGHT', 'DOWN', 'LEFT', 'UP'][self.search_direction]}",
                f"Search Clicks: {self.clicks_in_current_direction}/{self.max_clicks_per_direction}",
                f"Search Allowed: {'YES' if self.should_perform_search() else 'NO'}",
                f"No Target Frames: {self.consecutive_no_targets}/{self.search_trigger_threshold}",
                f"Static Loot: {'ACTIVE' if self.last_known_monster_position else 'INACTIVE'}",
                f"Post-Lock Wait: {'ACTIVE' if self.post_lock_wait_frames < self.max_post_lock_wait_frames else 'INACTIVE'}",
                f"Portal Escape: {'ACTIVE' if self.portal_avoidance_in_progress else 'INACTIVE'}",
                f"Movement History: {len(self.target_movement_history)} objects",
                f"Time Since Last F9: {int(time.time() - self.last_f9_press_time)}s",
                f"Time Since Player Moved: {int(time.time() - self.last_player_movement_time)}s" if self.last_player_position else "Player: Not Tracked",
                "",
                "=== ACTIVE TIMERS ===",
            ]

            timer_y_pos = len(info_lines) * 25 + 40
            for timer_name, progress in self.active_timers.items():
                timer_label = f"{timer_name}: {progress*100:.0f}%"
                cv2.putText(info_panel, timer_label, (15, timer_y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                bar_width = 250
                bar_height = 20
                bar_x = 300
                bar_y = timer_y_pos - 15

                cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

                progress_width = int(bar_width * progress)
                cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 200, 0), -1)

                cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                
                timer_y_pos += 35

            info_lines.extend([
                "",
                f"Auto F7 Presses: {self.detection_stats['auto_f7_presses']}",
                f"Auto F8 Presses: {self.detection_stats['auto_f8_presses']}",
                "Controls:",
                "F2: Pause/Resume",
                "D: Toggle Debug",
                "ESC: Quit"
            ])

            for i, line in enumerate(info_lines):
                if i >= len(info_lines) - 5:  
                    y_pos = timer_y_pos + (i - (len(info_lines) - 5)) * 25
                else:
                    y_pos = 40 + i * 25
                    
                color = (255, 255, 255)
                if "FARM" in line and "TARGETS" in line:
                    color = (0, 255, 0) 
                elif "CLOSE" in line and "TARGETS" in line:
                    color = (0, 165, 255) 
                elif "MONSTER" in line:
                    color = (255, 0, 0)
                elif "LOCKED" in line and self.locked_target_id:
                    color = (0, 255, 255)
                elif "PORTAL ESCAPES" in line and self.detection_stats['portal_emergency_escapes'] > 0:
                    color = (255, 0, 255)
                elif "BLACK TILE FORCED" in line and self.detection_stats['black_tile_forced_directions'] > 0:
                    color = (255, 165, 0)
                elif "PLAYER BBOX" in line and self.detection_stats['player_bbox_adjustments'] > 0:
                    color = (255, 0, 0) 
                elif "MOVEMENT PREDICTIONS" in line and self.detection_stats['movement_predictions'] > 0:
                    color = (0, 255, 0)  
                elif "PREDICTED TARGETS HIT" in line and self.detection_stats['predicted_targets_hit'] > 0:
                    color = (0, 255, 0) 
                elif "STABLE LOCKS" in line and self.detection_stats['stable_locks_maintained'] > 0:
                    color = (0, 255, 0)  
                elif "UNNECESSARY SWITCHES" in line and self.detection_stats['unnecessary_switches_prevented'] > 0:
                    color = (0, 255, 0)  
                elif "AUTO F9 PRESSES" in line and self.detection_stats['auto_f9_presses'] > 0:
                    color = (0, 255, 0)  
                elif "SEARCH ALLOWED: NO" in line:
                    color = (0, 0, 255)
                elif "SEARCH ALLOWED: YES" in line:
                    color = (0, 255, 0)
                elif "NO TARGET FRAMES" in line and self.consecutive_no_targets >= self.search_trigger_threshold:
                    color = (0, 255, 0)
                elif "FPS" in line and self.detection_stats['current_fps'] < 60:
                    color = (0, 0, 255)
                elif "FPS" in line and self.detection_stats['current_fps'] >= 90:
                    color = (0, 255, 0)
                elif "LOOT FOUND" in line and self.detection_stats['loot_found'] > 0:
                    color = (0, 255, 0)
                elif "STUCK RESETS" in line and self.detection_stats['player_stuck_resets'] > 0:
                    color = (255, 0, 0)
                elif "DISTANCE VIOLATIONS" in line and self.detection_stats['search_distance_violations'] > 0:
                    color = (255, 0, 0)
                elif "DYNAMIC ADJUSTMENTS" in line and self.detection_stats['dynamic_loot_adjustments'] > 0:
                    color = (0, 255, 255)
                elif "PORTAL AVOIDANCES" in line and self.detection_stats['portal_avoidances'] > 0:
                    color = (255, 0, 255)
                elif "BLACK TILE AVOIDANCES" in line and self.detection_stats['black_tile_avoidances'] > 0:
                    color = (0, 0, 0)
                elif "POST-LOCK TARGETS" in line and self.detection_stats['post_lock_wait_targets'] > 0:
                    color = (255, 165, 0)
                elif "STATIC LOOT: ACTIVE" in line:
                    color = (0, 255, 255)
                elif "POST-LOCK WAIT: ACTIVE" in line:
                    color = (255, 165, 0)
                elif "PORTAL ESCAPE: ACTIVE" in line:
                    color = (255, 0, 255)
                    
                cv2.putText(info_panel, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            debug_display = cv2.resize(debug_frame, 
                                     (int(self.DETECTION_WIDTH * self.debug_scale), 
                                      int(self.DETECTION_HEIGHT * self.debug_scale)))
            info_display = cv2.resize(info_panel, 
                                    (600, int(self.DETECTION_HEIGHT * self.debug_scale)))
            
            combined_display = np.hstack([debug_display, info_display])
            cv2.imshow(self.debug_window_name, combined_display)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Debug display error: {e}")

    def toggle_pause(self):
        global is_paused
        is_paused = not is_paused
        status = "PAUSED" if is_paused else "RESUMED"
        print(f"\n=== {status} ===\n")

    def toggle_debug_monitor(self):
        self.debug_monitor_enabled = not self.debug_monitor_enabled
        status = "ENABLED" if self.debug_monitor_enabled else "DISABLED"
        print(f"Debug monitor {status}")

    def setup_hotkeys(self):
        keyboard.add_hotkey(self.PAUSE_KEY, self.toggle_pause)
        keyboard.add_hotkey('d', self.toggle_debug_monitor)
        print(f"Pause: {self.PAUSE_KEY.upper()}, Debug: D")

    def run(self):
        global keep_running, is_paused
        
        self.setup_hotkeys()
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        pyautogui.press('f4')
        print("⏰ Initial F4 press - starting 3-minute timer")
        self.last_f9_press_time = time.time()  

        pyautogui.press('f7')
        print("⏰ Initial F7 press - starting 2-minute timer")
        self.last_f7_press_time = time.time()

        pyautogui.press('f8')  
        print("⏰ Initial F8 press - starting 30-minute timer")
        self.last_f8_press_time = time.time()
        
        frame_time = 1.0 / self.PROCESSING_FPS
        
        try:
            while keep_running:
                if is_paused:
                    time.sleep(0.1)
                    continue
                
                start_time = time.time()

                self.auto_press_f9()
                self.auto_press_f7_f8()

                frame = self.cpp.capture_region(
                    self.DETECTION_LEFT, self.DETECTION_TOP, 
                    self.DETECTION_WIDTH, self.DETECTION_HEIGHT
                )
                
                if frame is None:
                    try:
                        screenshot = pyautogui.screenshot(region=(
                            self.DETECTION_LEFT, self.DETECTION_TOP, 
                            self.DETECTION_WIDTH, self.DETECTION_HEIGHT
                        ))
                        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                    except:
                        continue
                    
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                detected_objects = self.detector.detect_objects(frame)

                detected_objects = self.adjust_player_bounding_box(detected_objects)

                tracked_objects = self.object_tracker.update(detected_objects)

                self.update_target_movement_history(tracked_objects)

                portal_avoidance_active = self.avoid_portal(tracked_objects)
                if portal_avoidance_active:
                    processing_time = time.time() - start_time
                    sleep_time = max(0, frame_time - processing_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                self.detect_player_movement(tracked_objects)

                target = self.find_best_target(tracked_objects)
                
                if target:
                    self.consecutive_no_targets = 0
                    target_x, target_y = target['screen_center']

                    dx = target_x - self.center_x
                    dy = target_y - self.center_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= self.close_target_radius and self.close_target_constant_click:
                        self.click_close_target_continuously(target_x, target_y)
                    else:
                        self.click_target_immediately(target_x, target_y)
                else:
                    self.consecutive_no_targets += 1
                    if self.consecutive_no_targets >= self.search_trigger_threshold and self.should_perform_search():
                        self.perform_smooth_search()
                
                processing_time = time.time() - start_time
                self.create_debug_display(frame, tracked_objects, processing_time)
                sleep_time = max(0, frame_time - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            keep_running = False
            try:
                cv2.destroyAllWindows()
            except:
                pass
            print("Bot stopped.")

if __name__ == "__main__":
    MODEL_PATH = 'C:/Users/python_app/runs/best.pt'
    bot = OptimizedMonsterBot(MODEL_PATH)

    bot.run()
