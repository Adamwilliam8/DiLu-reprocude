# -*- coding: utf-8 -*-
"""
Created on 2025/10/26

@author: Adam
"""

import os
import yaml
import torch
import numpy as np
from typing import Union
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

### 导入 配置文件config.yaml
current_dir = os.path.dirname(os.path.abspath(__file__))  # dilu/model/
parent_dir = os.path.dirname(current_dir)                  # dilu/
project_root = os.path.dirname(parent_dir)                 # DiLu-main/
config_path = os.path.join(project_root, "config.yaml")
OPENAI_CONFIG = yaml.load(open(config_path), Loader=yaml.FullLoader)


class CollisionPredictor:
    def __init__(self, use_llm=False, threshold=0.5, device='cpu'):
        """
        碰撞预测器
        Args:
            use_llm: 是否使用 LLM 进行预测（True）或使用简单规则（False）
            threshold: 碰撞概率阈值
            device: 设备（'cpu' 或 'cuda'）
        """
        self.use_llm = use_llm
        self.threshold = threshold
        self.device = device
        
        if self.use_llm:
            self._init_llm()
    
    def _init_llm(self):
        """初始化 LLM 模型"""
        oai_api_type = OPENAI_CONFIG["OPENAI_API_TYPE"]
        
        if oai_api_type == "azure":
            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_CHAT_DEPLOY_NAME"),
                temperature=0.0,  # 使用较低温度保证稳定性
                max_tokens=100,
                request_timeout=30,
            )
        else:  # openai or deepseek
            self.llm = ChatDeepSeek(model=os.getenv("DEEPSEEK_MODEL"),
                                    temperature=os.getenv("DEEPSEEK_TEMPERATURE"),
                                    max_tokens=os.getenv("DEEPSEEK_MAX_TOKENS"),
                                    request_timeout=30)
    
    def _obs_to_description(self, obs: np.ndarray, action: int) -> str:
        """
        将观测数据转换为文本描述
        Args:
            obs: 观测数据，shape=(vehicle_count, 5)，特征为 [presence, x, y, vx, vy]
            action: 计划执行的动作
        Returns:
            场景文本描述
        """
        action_names = {
            0: 'Turn-left (change to left lane)',
            1: 'IDLE (maintain current lane and speed)',
            2: 'Turn-right (change to right lane)',
            3: 'Acceleration',
            4: 'Deceleration'
        }
        
        # ego 车辆（第一行）
        ego = obs[0]
        ego_x, ego_y, ego_vx, ego_vy = ego[1], ego[2], ego[3], ego[4]
        ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
        
        description = f"### Current Situation:\n"
        description += f"- **Ego Vehicle**: Position ({ego_x:.1f}, {ego_y:.1f}), Speed {ego_speed:.1f} m/s\n"
        description += f"- **Planned Action**: {action_names.get(action, 'Unknown')}\n\n"
        
        description += "### Surrounding Vehicles:\n"
        
        # 周围车辆
        nearby_vehicles = []
        for i in range(1, len(obs)):
            vehicle = obs[i]
            if vehicle[0] < 0.5:  # presence = 0，车辆不存在
                continue
            
            v_x, v_y, v_vx, v_vy = vehicle[1], vehicle[2], vehicle[3], vehicle[4]
            v_speed = np.sqrt(v_vx**2 + v_vy**2)
            
            # 计算相对位置
            rel_x = v_x - ego_x
            rel_y = v_y - ego_y
            distance = np.sqrt(rel_x**2 + rel_y**2)
            
            # 判断方向
            if abs(rel_y) < 2.0:  # 同一车道
                if rel_x > 0:
                    position = "ahead in same lane"
                else:
                    position = "behind in same lane"
            elif rel_y > 0:  # 左侧
                position = "on the left"
            else:  # 右侧
                position = "on the right"
            
            nearby_vehicles.append({
                'distance': distance,
                'position': position,
                'speed': v_speed,
                'rel_x': rel_x
            })
        
        # 按距离排序，只保留最近的 5 辆车
        nearby_vehicles.sort(key=lambda x: x['distance'])
        for idx, v in enumerate(nearby_vehicles[:5], 1):
            description += f"{idx}. Vehicle {v['position']}, Distance {v['distance']:.1f}m, Speed {v['speed']:.1f} m/s\n"
        
        if not nearby_vehicles:
            description += "No nearby vehicles detected.\n"
        
        return description
    
    def _construct_prompt(self, scene_description: str) -> list:
        """构造 LLM 提示词"""
        system_msg = SystemMessage(content="""You are a collision risk assessment expert for autonomous driving.
Your task is to evaluate the collision probability of a planned driving action based on the current traffic situation.

**Output Format (IMPORTANT):**
You must respond with ONLY a single number between 0.0 and 1.0, representing the collision probability.
- 0.0 means completely safe (no collision risk)
- 1.0 means imminent collision (certain to crash)
- Values in between represent varying levels of risk

**Assessment Criteria:**
- Distance to nearby vehicles (closer = higher risk)
- Relative speed differences (large difference = higher risk)
- Planned action type (lane changes are riskier than maintaining lane)
- Vehicle positions (vehicles directly ahead/behind are more critical)

Examples:
- If a vehicle is 50m ahead and ego is accelerating: output "0.1"
- If a vehicle is 5m ahead and ego is accelerating: output "0.9"
- If changing lanes with a vehicle 3m on the side: output "0.85"
- If maintaining lane with no nearby vehicles: output "0.05"

Remember: Output ONLY the probability number, nothing else.""")
        
        human_msg = HumanMessage(content=f"""{scene_description}

Based on the above situation, what is the collision probability if the ego vehicle executes the planned action?
Output only a number between 0.0 and 1.0:""")
        
        return [system_msg, human_msg]
    
    def predict_prob(self, obs: np.ndarray, action: int) -> float:
        """
        预测碰撞概率
        Args:
            obs: 观测数据
            action: 计划执行的动作
        Returns:
            碰撞概率 (0.0 - 1.0)
        """
        if not self.use_llm:
            # 简单规则：基于最近车辆距离
            return self._rule_based_prediction(obs, action)
        
        try:
            # LLM 预测
            scene_desc = self._obs_to_description(obs, action)
            messages = self._construct_prompt(scene_desc)
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # 解析 LLM 输出
            try:
                # 提取数字（处理可能的格式变化）
                import re
                numbers = re.findall(r'\d+\.?\d*', response_text)
                if numbers:
                    prob = float(numbers[0])
                    # 确保在 [0, 1] 范围内
                    prob = max(0.0, min(1.0, prob))
                    return prob
                else:
                    print(f"[yellow]Warning: LLM output invalid, using rule-based fallback[/yellow]")
                    return self._rule_based_prediction(obs, action)
            except ValueError:
                print(f"[yellow]Warning: Cannot parse LLM output '{response_text}', using rule-based fallback[/yellow]")
                return self._rule_based_prediction(obs, action)
        
        except Exception as e:
            print(f"[red]Error in LLM collision prediction: {e}[/red]")
            return self._rule_based_prediction(obs, action)
    
    def _rule_based_prediction(self, obs: np.ndarray, action: int) -> float:
        """基于规则的碰撞预测（备用方案）"""
        ego = obs[0]
        ego_x, ego_y = ego[1], ego[2]
        
        min_distance = float('inf')
        
        # 找到最近车辆
        for i in range(1, len(obs)):
            if obs[i][0] < 0.5:  # 车辆不存在
                continue
            
            v_x, v_y = obs[i][1], obs[i][2]
            distance = np.sqrt((v_x - ego_x)**2 + (v_y - ego_y)**2)
            min_distance = min(min_distance, distance)
        
        # 简单映射：距离越近，概率越高
        if min_distance == float('inf'):
            return 0.05  # 无车辆
        elif min_distance < 5:
            return 0.9
        elif min_distance < 10:
            return 0.6
        elif min_distance < 20:
            return 0.3
        else:
            return 0.1