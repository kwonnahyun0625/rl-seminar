# rl-seminar
## 📃 사전 준비

## ✨ 목표 ✨
심층 강화학습 기반 사족 보행 로봇 제어의 경험을 쌓고, Isaac gym을 이용한 강화학습 프레임워크와 친해지기 😀
#### 🏁 To do list
실습1. legged gym 기반 강화학습 환경 설치하기 </br>
실습2. action, reward, observation 수정해보기 </br>
실습3. 새로운 로봇을 이용한 강화학습  </br>
실습4. Walk These Ways </br>
실습5. legged gym학습 환경에 adaptation module 추가해서 속도 추정기 만들기 </br>
실습6. 학습 모델의 sim to real(sim)을 위한 lcm 실습 </br>
실습7. wtw의 deploy코드와 가제보 시뮬레이션을 이용한 sim to sim환경 만들기 </br>

## 🔴 실습1. legged gym 기반 강화학습 환경 설치하기
```
## 가상환경 생성 및 pytorch설치
>> conda create -n seminar python=3.8
>> conda activate seminar
>> pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

## isaac gym 설치
>> cd isaacgym/python/
>> pip install -e .

## isaac gym 설치 테스트
>> cd examples/
>> python 1080_balls_of_solitude.py

## rsl_rl install
>> git clone https://github.com/leggedrobotics/rsl_rl.git
>> cd rsl_rl && git checkout v1.0.2 && pip install -e .

## legged_gym install
>> git clone https://github.com/leggedrobotics/legged_gym.git
>> cd legged_gym && pip install -e .

## train & play
>> cd legged_gym/scripts/
>> python train.py --task=anymal_c_flat
>> python play.py --task=anymal_c_flat
```

```
## error

오류#1 ModuleNotFoundError: No module named 'tensorboard'
해결 >> pip install tensorboard

오류#2 AttributeError: module 'distutils' has no attribute 'version'
해결 >> pip install future
    >> pip install --upgrade torch
    
오류#3 The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
해결 >> pip install "numpy<1.24"

오류#4 AttributeError: module 'numpy' has no attribute 'float'.
해결 >> pip install "numpy<1.24"

오류#5 The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
해결 >> pip install "numpy<1.24"

오류#6 RuntimeError: Ninja is required to load C++ extensions
해결 >> wget https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip
    >> sudo unzip ninja-linux.zip -d /usr/local/bin/
    >> sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force  
    
오류#7 RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch) 
해결 >> pip3 uninstall torch torchvision torchaudio
    >> pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu123/torch_stable.html
```
🔎 참고자료
1. legged gym repo : https://github.com/leggedrobotics/legged_gym
2. rsl_rl gym repo : https://github.com/leggedrobotics/rsl_rl

## 🟠 실습2. action, reward, observation 수정해보기 
## 🟡 실습3. 새로운 로봇을 이용한 강화학습 
## 🟢 실습4. Walk These Ways
```
## 가상환경 생성 및 pytorch설치
>> conda create -n seminar2 python=3.8
>> conda activate seminar2
>> pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

## isaac gym 설치
>> cd isaacgym/python/
>> pip install -e .

## isaac gym 설치 테스트
>> cd examples/
>> python 1080_balls_of_solitude.py

## walk-these-ways install
>> git clone https://github.com/Improbable-AI/walk-these-ways.git
>> pip install -e .

## train & play
>> cd scripts/
>> python train.py 
>> python play.py 
```

```
## error

오류#1 RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch) 
해결 >> pip3 uninstall torch torchvision torchaudio
    >> pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu123/torch_stable.html
```
## 🔵 실습5. legged gym학습 환경에 adaptation module 추가해서 속도 추정기 만들기
#### 1️⃣ legged_gym 수정
◾ legged_robot.py
```p
## def compute_observations 
self.obs_buf = torch.cat((  #self.base_lin_vel * self.obs_scales.lin_vel,
                            #self.base_ang_vel  * self.obs_scales.ang_vel,
                            self.projected_gravity,
                            self.commands[:, :3] * self.commands_scale,
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                            self.dof_vel * self.obs_scales.dof_vel,
                            self.actions
                            ),dim=-1)
##############################################################################
self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
self.privileged_obs_buf = torch.cat((self.obs_buf,
                                     self.base_lin_vel * self.obs_scales.lin_vel,
                                     self.base_ang_vel  * self.obs_scales.ang_vel ),dim=-1)

## def _get_noise_scale_vec
noise_vec[0:3] = noise_scales.gravity * noise_level
noise_vec[3:6] = 0. # commands
noise_vec[6:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
noise_vec[18:30] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
noise_vec[30:42] = 0. # previous actions
``` 

◾ anymal_c_flat_config.py
```p
## class env( AnymalCRoughCfg.env ):
num_observations = 42
num_privileged_obs = 48
```
#### 2️⃣ rsl_rl 수정
◾ actor_critic.py
```p
## def __init__
# Adaptation module
adaptation_module_branch_hidden_dims = [256, 128]
num_latent = 6
mlp_input_dim_a = num_actor_obs + num_latent

adaptation_module_layers = []
adaptation_module_layers.append(nn.Linear(num_actor_obs, adaptation_module_branch_hidden_dims[0]))
adaptation_module_layers.append(activation)
for l in range(len(adaptation_module_branch_hidden_dims)):
    if l == len(adaptation_module_branch_hidden_dims) - 1:
        adaptation_module_layers.append(
            nn.Linear(adaptation_module_branch_hidden_dims[l], num_latent))
    else:
        adaptation_module_layers.append(
            nn.Linear(adaptation_module_branch_hidden_dims[l],
                      adaptation_module_branch_hidden_dims[l + 1]))
        adaptation_module_layers.append(activation)
self.adaptation_module = nn.Sequential(*adaptation_module_layers)
print(f"Adaptation Module: {self.adaptation_module}")

## def update_distribution
latent = self.adaptation_module(observations)
mean = self.actor(torch.cat((observations, latent), dim=-1))

## def act_inference
latent = self.adaptation_module(observations)
actions_mean = self.actor(torch.cat((observations, latent), dim=-1))

```

◾ ppo.py
```p
## def __init__
self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

## def update
mean_adaptation_module_loss = 0
##############################################################################
# Adaptation module gradient step
adaptation_pred = self.actor_critic.adaptation_module(obs_batch)
with torch.no_grad():
    adaptation_target = critic_obs_batch[:, 42:48]
selection_indices = torch.linspace(0, adaptation_pred.shape[1]-1, steps=adaptation_pred.shape[1], dtype=torch.long)
adaptation_loss = F.mse_loss(adaptation_pred[:, selection_indices], adaptation_target[:, selection_indices])
self.adaptation_module_optimizer.zero_grad()
adaptation_loss.backward()
self.adaptation_module_optimizer.step()
mean_adaptation_module_loss += adaptation_loss.item()
##############################################################################
import torch.nn.functional as F
##############################################################################
mean_adaptation_module_loss /= num_updates
##############################################################################
return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss

```

◾ on_policy_runner.py
```p
## def learn
mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss = self.alg.update()

##
self.writer.add_scalar('Loss/adaptation_module', locs['mean_adaptation_module_loss'], locs['it'])
##############################################################################
f"""{'adaptation_module loss:':>{pad}} {locs['mean_adaptation_module_loss']:.4f}\n"""
##############################################################################
f"""{'adaptation_module loss:':>{pad}} {locs['mean_adaptation_module_loss']:.4f}\n""" 
```

## 🟣 실습6. 학습 모델의 sim to real(sim)을 위한 lcm 실습
## 🟤 실습7. wtw의 deploy코드와 가제보 시뮬레이션을 이용한 sim to sim환경 만들기
