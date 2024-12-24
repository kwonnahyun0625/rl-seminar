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
## 🟣 실습6. 학습 모델의 sim to real(sim)을 위한 lcm 실습
## 🟤 실습7. wtw의 deploy코드와 가제보 시뮬레이션을 이용한 sim to sim환경 만들기
