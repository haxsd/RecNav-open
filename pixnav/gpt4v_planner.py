import copy
import numpy as np
import os
from llm_utils.gpt_request import gptv_response, gpt_response
from llm_utils.nav_prompt import GPT4V_PROMPT
from cv_utils.detection_tools import *
from cv_utils.segmentation_tools import *
import cv2
import ast
class GPT4V_Planner:
    def __init__(self,dino_model,sam_model,save_monitor_image=False,image_scale=0.5,keep_debug_data=False):
        self.gptv_trajectory = []
        self.dino_model = dino_model
        self.sam_model = sam_model
        self.detect_objects = ['bed','sofa','chair','plant','tv','toilet','floor']
        self.save_monitor_image = save_monitor_image
        self.image_scale = image_scale
        self.keep_debug_data = keep_debug_data
        self.verbose = str(os.environ.get("PIXNAV_PLANNER_VERBOSE", "0")).strip().lower() in {"1", "true", "yes", "on"}
        try:
            self.vision_retries = max(1, int(os.environ.get("PIXNAV_VISION_RETRIES", "3")))
        except ValueError:
            self.vision_retries = 3
    
    def reset(self,object_goal):
        # translation to align for the detection model
        if object_goal == 'tv_monitor':
            self.object_goal = 'tv'
        else:
            self.object_goal = object_goal

        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        self.direction_image_trajectory = []
        self.direction_mask_trajectory = []

    def concat_panoramic(self,images,angles):
        try:
            height,width = images[0].shape[0],images[0].shape[1]
        except:
            height,width = 480,640
        background_image = np.zeros((2*height + 3*10, 3*width + 4*10,3),np.uint8)
        copy_images = np.array(images,dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
               continue
            copy_images[i] = cv2.putText(copy_images[i],"Angle %d"%angles[i],(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i//2) % 3
            background_image[10*(row+1)+row*height:10*(row+1)+row*height+height:,10*(col+1)+col*width:10*(col+1)+col*width+width,:] = copy_images[i]
        return background_image
    
    def _materialize_plan(self, direction_image, direction, goal_flag, return_debug_image):
        if self.dino_model is None or self.sam_model is None:
            height, width = direction_image.shape[:2]
            pixel_x, pixel_y = width // 2, height // 2
            debug_image = None
            if return_debug_image:
                debug_image = np.array(direction_image)
                debug_image = cv2.rectangle(debug_image,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,0,0),-1)
            debug_mask = np.zeros((height,width),np.uint8)
            debug_mask = cv2.rectangle(debug_mask,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,255,255),-1)
            if self.keep_debug_data:
                self.direction_image_trajectory.append(direction_image)
                self.direction_mask_trajectory.append(debug_mask)
            return direction_image,debug_mask,debug_image,direction,False

        direction_rgb = cv2.cvtColor(direction_image, cv2.COLOR_BGR2RGB)
        target_bbox = openset_detection(direction_rgb, self.detect_objects, self.dino_model)
        goal_class_idx = self.detect_objects.index(self.object_goal)
        if goal_class_idx not in target_bbox.class_id:
            goal_flag = False

        if goal_flag:
            # Reuse first detection: filter to goal-class boxes only (skip 2nd DINO call)
            _goal_indices = np.where(target_bbox.class_id == goal_class_idx)[0]
            if len(_goal_indices) > 0:
                bbox = copy.copy(target_bbox)
                bbox.xyxy = target_bbox.xyxy[_goal_indices]
                bbox.confidence = target_bbox.confidence[_goal_indices]
                bbox.class_id = target_bbox.class_id[_goal_indices]
            else:
                bbox = openset_detection(direction_rgb, [self.object_goal], self.dino_model)
        else:
            bbox = openset_detection(direction_rgb, ['floor'], self.dino_model)
        try:
            mask = sam_masking(direction_image,bbox.xyxy,self.sam_model)
        except:
            mask = np.ones_like(direction_image).mean(axis=-1)
        
        if self.keep_debug_data:
            self.direction_image_trajectory.append(direction_image)
            self.direction_mask_trajectory.append(mask)

        debug_mask = np.zeros(direction_image.shape[:2], np.uint8)
        pixel_y,pixel_x = np.where(mask>0)[0:2]
        pixel_y = int(pixel_y.mean())
        pixel_x = int(pixel_x.mean())
        debug_image = None
        if return_debug_image:
            debug_image = np.array(direction_image)
            debug_image = cv2.rectangle(debug_image,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,0,0),-1)
        debug_mask = cv2.rectangle(debug_mask,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,255,255),-1)
        return direction_image,debug_mask,debug_image,direction,goal_flag

    def make_plan(self,pano_images,return_debug_image=True):
        direction,goal_flag = self.query_gpt4v(pano_images)
        direction_image = pano_images[direction]
        return self._materialize_plan(direction_image, direction, goal_flag, return_debug_image)

    def make_plan_from_direction(self, pano_images, direction, goal_flag=False, return_debug_image=True):
        direction_idx = int(direction) % len(pano_images)
        direction_image = pano_images[direction_idx]
        return self._materialize_plan(direction_image, direction_idx, bool(goal_flag), return_debug_image)
        
    def query_gpt4v(self,pano_images):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = cv2.cvtColor(self.concat_panoramic(pano_images,angles),cv2.COLOR_BGR2RGB)

        h, w = inference_image.shape[:2]
        scaled_w = max(1, int(w * self.image_scale))
        scaled_h = max(1, int(h * self.image_scale))
        inference_image_small = cv2.resize(inference_image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        if self.save_monitor_image:
            cv2.imwrite("monitor-panoramic.jpg",inference_image_small)
        text_content = "<Target Object>:{}\n".format(self.object_goal)
        if self.keep_debug_data:
            self.gptv_trajectory.append("\nInput:\n%s \n"%text_content)
            self.panoramic_trajectory.append(inference_image)
        raw_answer = "Planner unavailable; fallback to random direction."
        answer = None
        last_error = ""
        for i in range(self.vision_retries):
            try:
                raw_answer = gptv_response(text_content,inference_image_small,GPT4V_PROMPT)
                if self.verbose:
                    print("GPT-4V Output Response: %s"%raw_answer)
                answer = raw_answer
                answer = answer[answer.index("{"):answer.index("}")+1]
                answer = ast.literal_eval(answer)
                if 'Reason' in answer.keys() and 'Angle' in answer.keys():
                    break
                assert answer['Angle'] in angles
            except Exception as exc:
                last_error = str(exc)
                if "image_url" in last_error or "unknown variant" in last_error:
                    break
                continue
        if answer is None:
            answer = self.query_text_planner(pano_images, angles)
            if answer is not None:
                raw_answer = str(answer)
            elif last_error:
                raw_answer = last_error
        if self.keep_debug_data:
            self.gptv_trajectory.append("GPT-4V Answer:\n%s"%raw_answer)
            self.panoramic_trajectory.append(inference_image)
        try:
            return (int(answer['Angle']//30))%12,answer['Flag']
        except:
            return 0, False

    def query_text_planner(self, pano_images, angles):
        if self.dino_model is None:
            return None
        summary_lines = []
        for idx, (image, angle) in enumerate(zip(pano_images, angles)):
            try:
                det = openset_detection(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.detect_objects, self.dino_model)
                entries = []
                for cid, conf in zip(det.class_id.tolist(), det.confidence.tolist()):
                    if cid is None:
                        continue
                    label = self.detect_objects[int(cid)]
                    entries.append(f"{label}:{float(conf):.3f}")
                if entries:
                    summary_lines.append(f"Angle {int(angle)}: " + ", ".join(entries[:5]))
                else:
                    summary_lines.append(f"Angle {int(angle)}: none")
            except Exception:
                summary_lines.append(f"Angle {int(angle)}: error")

        prompt = (
            f"Target object: {self.object_goal}\n"
            "Choose one angle from this list and whether target is likely visible.\n"
            "Return strict Python dict only, e.g. {'Angle': 90, 'Flag': True}.\n"
            + "\n".join(summary_lines)
        )
        try:
            raw = gpt_response(prompt, "")
            parsed = raw[raw.index("{"):raw.index("}")+1]
            parsed = ast.literal_eval(parsed)
            if "Angle" in parsed and parsed["Angle"] in angles:
                if "Flag" not in parsed:
                    parsed["Flag"] = False
                return parsed
        except Exception:
            return None
        return None
