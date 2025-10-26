############################################################################
## 把 感知模块/LLM输出解析给省了
#############################################################################
import copy
import random
import numpy as np
import yaml
import os
import getpass
import torch
from rich import print

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from dilu.scenario.envScenario import EnvScenario
from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.vectorStore import DrivingMemory
from dilu.driver_agent.reflectionAgent import ReflectionAgent
from dilu.model.PPO import PPOAgent,PPOConfig
from dilu.model.discrimination import AttentionDiscriminator
from dilu.model.collision_predictor import CollisionPredictor
from dilu.model.discrimination import DiscriminatorBuffer

from dilu.utils import imitation_reward, linear_eps

test_list_seed = [5838, 2421, 7294, 9650, 4176, 6382, 8765, 1348,
                  4213, 2572, 5678, 8587, 512, 7523, 6321, 5214, 31]

# os.environ 表示当前操作系统的环境变量
def setup_env(config):
    if config['OPENAI_API_TYPE'] == 'azure':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_VERSION"] = config['AZURE_API_VERSION']
        os.environ["OPENAI_API_BASE"] = config['AZURE_API_BASE']
        os.environ["OPENAI_API_KEY"] = config['AZURE_API_KEY']
        os.environ["AZURE_CHAT_DEPLOY_NAME"] = config['AZURE_CHAT_DEPLOY_NAME']
        os.environ["AZURE_EMBED_DEPLOY_NAME"] = config['AZURE_EMBED_DEPLOY_NAME']
    elif config['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
        os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
    elif config['OPENAI_API_TYPE'] == 'deepseek':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass(config['DEEPSEEK_KEY'])
        os.environ["DEEPSEEK_MODEL"] = config['DEEPSEEK_MODEL']
        os.environ["DEEPSEEK_BASE_URL"] = config['DEEPSEEK_BASE_URL']
        os.environ["DEEPSEEK_TEMPERATURE"] = config['DEEPSEEK_TEMPERATURE']
        os.environ["DEEPSEEK_MAX_TOKENS"] = config['DEEPSEEK_MAX_TOKENS']
    else:
        raise ValueError("Unknown OPENAI_API_TYPE, should be azure or openai")

    # environment setting
    env_config = {
        'highway-v0':
        {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": config["vehicle_count"],
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(5, 32, 9),
            },
            "lanes_count": 4,
            "other_vehicles_type": config["other_vehicle_type"],
            "duration": config["simulation_duration"],
            "vehicles_density": config["vehicles_density"],
            "show_trajectories": True,
            "render_agent": True,
            "scaling": 5,
            'initial_lane_id': None,
            "ego_spacing": 4,
        }
    }

    return env_config


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore") 

    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    env_config = setup_env(config)

    REFLECTION = config["reflection_module"]
    memory_path = config["memory_path"]
    few_shot_num = config["few_shot_num"]
    result_folder = config["result_folder"]
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_folder + "/" + 'log.txt', 'w') as f:
        f.write("memory_path {} | result_folder {} | few_shot_num: {} | lanes_count: {} \n".format(
            memory_path, result_folder, few_shot_num, env_config['highway-v0']['lanes_count']))

    # 记忆模块，加载记忆库
    agent_memory = DrivingMemory(db_path=memory_path)
    # 如有反思模块，则加载新记忆库用于更新
    if REFLECTION:
        updated_memory = DrivingMemory(db_path=memory_path + "_updated")
        updated_memory.combineMemory(agent_memory)

    ### 生成PPO智能体
    env = gym.make('highway-v0', render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    cfg = PPOConfig(
        policy_kwargs={"net_arch": [256, 256]},
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        device="cpu",
        tensorboard_log="../reproduce1/highway_ppo/",
        verbose=1,
    )
    agent = PPOAgent(state_dim, action_dim, cfg)

    ### Attention Discriminator
    disc = AttentionDiscriminator(
        state_dim, 
        action_dim, 
        config['discriminator']['hidden_sizes']
    )
    disc_opt = torch.optim.Adam(disc.parameters(), lr=1e-4)
    # 判别器缓冲区（用于批量更新）
    disc_buffer = DiscriminatorBuffer(capacity=10000)
    

    ### Collision Predictor (支持 LLM 或规则模式)
    coll_pred = CollisionPredictor(
        use_llm=config['collision_predictor']['use_llm'],
        threshold=config['collision_predictor']['threshold'],
        device=cfg.device
    )

    episode = 0
    global_step = 0  # 全局步数计数器
    while episode < config["episodes_num"]:  # 控制有几个cycle，每次重置环境
        ### setup highway-env
        envType = 'highway-v0'
        env = gym.make(envType, render_mode="rgb_array")
        env.configure(env_config[envType])
        result_prefix = f"highway_{episode}"
        env = RecordVideo(env, result_folder, name_prefix=result_prefix)
        env.unwrapped.set_record_video_wrapper(env)
        seed = random.choice(test_list_seed)
        obs, info = env.reset(seed=seed)
        env.render()

        # scenario and driver agent setting
        database_path = result_folder + "/" + result_prefix + ".db"
        sce = EnvScenario(env, envType, seed, database_path) # 基于highway-env的场景对象
        DA = DriverAgent(sce, verbose=True)  # 驾驶智能体对象
        if REFLECTION:
            RA = ReflectionAgent(verbose=True)

        response = "Not available"
        LLM_action = "Not available"
        docs = []
        collision_frame = -1

        try:
            already_decision_steps = 0
            for i in range(0, config["simulation_duration"]): # 控制每个cycle里环境跑多少步
                obs = np.array(obs, dtype=float)

                print("[cyan]Retreive similar memories...[/cyan]")
                ### 从记忆库中检索few_shot_num条与当前场景最相似的记忆
                fewshot_results = agent_memory.retriveMemory(
                    sce, i, few_shot_num) if few_shot_num > 0 else []
                fewshot_messages = []
                fewshot_answers = []
                fewshot_actions = []
                for fewshot_result in fewshot_results:
                    fewshot_messages.append(
                        fewshot_result["human_question"])
                    fewshot_answers.append(fewshot_result["LLM_response"])
                    fewshot_actions.append(fewshot_result["action"])
                    mode_action = max(
                        set(fewshot_actions), key=fewshot_actions.count)
                    mode_action_count = fewshot_actions.count(mode_action)
                if few_shot_num == 0:
                    print("[yellow]Now in the zero-shot mode, no few-shot memories.[/yellow]")
                else:
                    print("[green4]Successfully find[/green4]", len(
                        fewshot_actions), "[green4]similar memories![/green4]")

                ### 驾驶智能体决策
                sce_descrip = sce.describe(i)
                avail_action = sce.availableActionsDescription()
                print('[cyan]Scenario description: [/cyan]\n', sce_descrip)
                LLM_action, response, human_question, fewshot_answer = DA.few_shot_decision(
                    scenario_description=sce_descrip, available_actions=avail_action,
                    previous_decisions=LLM_action,
                    fewshot_messages=fewshot_messages,
                    driving_intensions="Drive safely and avoid collisons",
                    fewshot_answers=fewshot_answers,
                )
                RL_action, logprob, value = agent.act(obs)
                ### 碰撞预测,评估该决策的碰撞概率
                rho = coll_pred.predict_prob(obs, LLM_action)
                print(f"[cyan]Collision probability: {rho:.2f}[/cyan]")
                ## 如果预测的碰撞概率过高（超过阈值） 
                ## 系统将_不会执行_LLM的决策，而是用安全托底策略_（如紧急制动、保持车道等）
                if rho > cfg['collision_predictor']['threshold']:
                    print(f"[red]High collision risk detected ({rho:.2f})! Overriding to IDLE action.[/red]")
                    LLM_action = 1  # 安全优先：高风险强制 IDLE
                
                docs.append({
                    "sce_descrip": sce_descrip,
                    "human_question": human_question,
                    "response": response,
                    "LLM_action": LLM_action,
                    "RL_action": RL_action,
                    "sce": copy.deepcopy(sce)
                })

                obs_next, reward, done, info, _ = env.step(RL_action)
                already_decision_steps += 1
                global_step += 1

                ### 生成模仿奖励,然后加给原奖励，这样LLM选择可以影响RL训练
                '''disc判别器模块要修改'''
                r_imit = config['discriminator']['beta'] * imitation_reward(
                    disc, obs, LLM_action, RL_action, config['discriminator']['eps_smooth']
                )
                r_total = reward + r_imit

                ### 判别器训练：使用 (s, a_ex) 作为 expert，(s, a_t) 作为 agent
                ## 存储判别器训练数据到缓冲区
                s_tensor = torch.tensor(obs, dtype=torch.float32)
                a_ex_tensor = torch.tensor(LLM_action, dtype=torch.int64)
                a_ag_tensor = torch.tensor(RL_action, dtype=torch.int64)
                disc_buffer.add(s_tensor, a_ex_tensor, a_ag_tensor)
                ## 判别器批量训练（每 4 步更新一次）
                if global_step % 4 == 0 and len(disc_buffer) >= 64:
                    batch_size = 64
                    s_ex, a_ex, s_ag, a_ag = disc_buffer.sample(batch_size)
                    # 计算判别器损失
                    d_ex = disc(s_ex, a_ex)  # 专家动作相似度
                    d_ag = disc(s_ag, a_ag)  # 智能体动作相似度
                    # 二元交叉熵损失：鼓励 D(expert)=1, D(agent)=0
                    loss_d = -(torch.log(d_ex + 1e-8).mean() + torch.log(1.0 - d_ag + 1e-8).mean())
                    # 更新判别器
                    disc_opt.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)  # 梯度裁剪
                    disc_opt.step()
                    if global_step % 100 == 0:
                        print(f"[magenta]Discriminator Loss: {loss_d.item():.4f} | D(expert): {d_ex.mean().item():.3f} | D(agent): {d_ag.mean().item():.3f}[/magenta]")

                ### PPO学习
                # **存储经验到 PPO 缓冲区（使用 r_total）**
                agent.store_transition(obs, RL_action, logprob, r_total, done, value)
                # **检查是否需要更新 PPO**
                if agent.is_ready_to_update():
                    agent.update_from_buffer(obs_next)
                obs = obs_next  # 更新观察

                env.render()
                sce.promptsCommit(i, None, done, human_question,
                                  fewshot_answer, response)
                env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()

                print("--------------------")

                # 如果碰撞了，做标记（highway环境只有碰撞了，done才会是True）
                if done:
                    print("[red]Simulation crash after running steps: [/red] ", i)
                    collision_frame = i
                    break
        finally:

            with open(result_folder + "/" + 'log.txt', 'a') as f:
                f.write(
                    "Simulation {} | Seed {} | Steps: {} | File prefix: {} \n".format(episode, seed, already_decision_steps, result_prefix))
                
            if REFLECTION:
                print("[yellow]Now running reflection agent...[/yellow]")
                if collision_frame != -1: # 如果碰撞了
                    for i in range(collision_frame, -1, -1):
                        if docs[i]["RL_action"] != 4:  # not decelearate(碰撞发生时，最后一步通常是“减速”)
                            corrected_response = RA.reflection(
                                docs[i]["human_question"], docs[i]["response"])
                            
                            choice = input("[yellow]Do you want to add this new memory item to update memory module? (Y/N): ").strip().upper()
                            if choice == 'Y':
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    corrected_response,
                                    docs[i]["RL_action"],
                                    docs[i]["sce"],
                                    comments="mistake-correction"
                                )
                                print("[green] Successfully add a new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                            else:
                                print("[blue]Ignore this new memory item[/blue]")
                            break
                else:  # 没碰撞的话，怎么处理记忆
                    print("[yellow]Do you want to add[/yellow]",len(docs)//5, "[yellow]new memory item to update memory module?[/yellow]",end="")
                    choice = input("(Y/N): ").strip().upper()
                    if choice == 'Y':
                        cnt = 0
                        storage_probability = config["storage_probability"]
                        for i in range(len(docs)):
                            # 每条记忆独立判断是否存储
                            if random.random() < storage_probability:
                                updated_memory.addMemory(
                                    docs[i]["sce_descrip"],
                                    docs[i]["human_question"],
                                    docs[i]["response"],
                                    docs[i]["RL_action"],
                                    docs[i]["sce"],
                                    comments="no-mistake-random"
                                )
                                cnt +=1
                        if cnt > 0:
                            print("[green] Successfully add[/green] ",cnt," [green]new memory item to update memory module.[/green]. Now the database has ", len(
                                    updated_memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
                        else:
                            print("[blue]No memories were stored this episode (by random chance)[/blue]")
                    else:
                        print("[blue]Ignore these new memory items[/blue]")
            

            print("==========Simulation {} Done==========".format(episode))
            episode += 1
            env.close()
