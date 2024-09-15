# dreamwaq


https://github.com/curieuxjy/dreamwaq/assets/40867411/5dcea5c9-3ff3-469d-baa7-70f0852a0395

[🎥 1080 Streaming Video in YouTube](https://youtu.be/5rwFcz-lerw)

## Index

- [Start Manual](https://github.com/curieuxjy/dreamwaq#start-manual): 프로젝트 환경 설정과 실행 방법에 대한 내용 
- [Main Code Structure](https://github.com/curieuxjy/dreamwaq#main-code-structure): 프로젝트 주요 코드 설명 
- [Result Graphs](https://github.com/curieuxjy/dreamwaq#result-graphs): 프로젝트 학습 결과 그래프
- [Result Motions](https://github.com/curieuxjy/dreamwaq#result-motions): 프로젝트 학습 결과 보행 모션 영상(Video 파트별 gif)

## Start Manual

### Start **w/o** this repository
> 이 repository와 상관없이 구현 프로젝트 초기 셋팅입니다.  이 repository를 기반으로 실행 하려면, 아래 w/ 실행 단계를 참고해주세요.  
1. IsaacGym ver.4 설치
2. [rsl-rl](https://github.com/leggedrobotics/rsl_rl) github에서 **zip**파일로 다운받아서 설치 `pip install -e .`
3. [legged-gym](https://github.com/leggedrobotics/legged_gym) github에서 **zip**파일로 다운받아서 설치 `pip install -e .`
4. wandb 등 몇가지 실험 로깅에 필요한 부분 수정(각자 계정으로 로그인해야 함) 

### Start **w/** this repository
> 이 repository를 기반으로 프로젝트를 시작할 때 아래와 같이 진행해주세요.  

1. IsaacGym ver.4 설치 [isaac-gym 페이지](https://developer.nvidia.com/isaac-gym)
2. `rsl-rl/` 위치에서 `pip install -e .`
3. `legged-gym/` 위치에서 `pip install -e .`
4. A1으로 Rough terrain locomotion learning 시작(아래 표 참고)

| option             | config           | critic_obs | actor_obs | memo                                               |
|--------------------|------------------|------------|-----------|:---------------------------------------------------|
| `--task=a1_base`   | A1RoughBaseCfg   | 45         | 45        | lin_vel을 뺀 observation                             |
| `--task=a1_oracle` | A1RoughOracleCfg | 238        | 238       | true_lin_vel + privileged(d,h)                     |
| `--task=a1_waq`    | A1RoughBaseCfg   | 238        | 64        | est_lin_vel + privileged / obs_history(timestep 5) |

### Start **w/** docker
> 이 repository를 기반으로 docker 를 통해 시작할 때 아래와 같이 진행해주세요.
> CUDA 12.1 이상을 지원하는 드라이버가 설치 되어있어야 합니다.

1. IsaacGym ver.4 다운로드 [isaac-gym 페이지](https://developer.nvidia.com/isaac-gym)
2. 다운로드 받은 `IsaacGym_Preview_4_Package.tar.gz` 파일을 `asset/IsaacGym_Preview_4_Package.tar.gz` 로 이동
3. 다음 명령어로 도커 빌드 `docker build . -t dreamwaq/dreamwaq -f docker/Dockerfile  --build-arg UID=$(id -u) --build-arg GID=$(id -g)`
4. 다음 명령어로 도커 실행 `docker run -ti --privileged -e DISPLAY=:0 -e TERM=xterm-256color -v /tmp/.X11-unix:/tmp/.X11-unix:ro --network host -v $PWD/dreamwaq:/home/user/dreamwaq --gpus all dreamwaq/dreamwaq /usr/bin/zsh`

### Command 

- training : `python train.py --task=[TASK_NAME] --headless`
  - `--headless`: 시뮬레이터 창을 띄우지않고 학습 실행하는 코드. display가 없는 서버에서 실행시 추가하는 option.
- inferencing : `python play.py --task=[TASK_NAME] --load_run=[LOAD_FOLDER] --checkpoint=[CHECKPOINT_NUMBER]`
  - `[LOAD_FOLDER]`: `legged_gym/logs/[task별 폴더]` 내부에 있는 파일 명. (예) `Sep04_14-24-54_waq`
    - `[task별 폴더]`: rough_a1/rough_a1_waq/rough_a1_est
  - `[CHECKPOINT_NUMBER]`: `[LOAD_FOLDER]`에 있는 **model_[NUMBER].pt** 파일의 번호. (예) `250`
  - 완성된 command (예) `python play.py --task=a1_waq --load_run=Sep04_14-24-54_waq --checkpoint=250`
  - 하나의 agent를 가까이서 보는 inferencing code: `mini_test.py` (옵션은 `play.py`와 동일)
  - 각 inferencing script에 main loop에 조절하는 옵션들이 있으니 참고해서 True/False 조정.
- 다른 컴퓨터에서 training 된 **model_[NUMBER].pt** 파일을 inferencing하고 싶다면,
  - TRAINING **{@computer_A}** | INFERENCING **{@computer_B}**
    1. {@computer_B} `legged_gym/logs/[task별 폴더]`에 `FOLDER_NAME`이라는 새로운 폴더를 만든 뒤,
    2. {@computer_B} `FOLDER_NAME`에 {@computer_A}의 **model_[NUMBER].pt 파일**을 copy&paste
    2. {@computer_B} `python play.py --task=[TASK_NAME] --load_run=[FOLDER_NAME] --checkpoint=[NUMEBR]` 로 실행.

## Main Code Structure


- 프로젝트 코드들 중 중요 파일들에 대한 설명입니다. 프로젝트에서 사용된 로봇 플랫폼과 알고리즘 위주의 코드들을 선정하였으며, 실행 파일명 옆에 있는 설명을 참고해주세요.
   - 사용 로봇 플랫폼(환경): A1
   - 사용 학습 알고리즘: PPO

```
dreamwaq
│
├── legged_gym
│   ├── legged_gym
│   │   ├── envs
│   │   │   ├── __init__.py: 학습 실행을 위한 환경 등록. task_registry에서 참조.
│   │   │   ├── a1/a1_config.py: A1 플랫폼에 맞는 변수 클래스. legged_robot_config.py의 클래스 상속.
│   │   │   └── base
│   │   │        ├── legged_robot.py: locomotion task를 위한 기본 환경 클래스. LeggedRobot Class 
│   │   │        └── legged_robot_config.py: LeggedRobot을 위한 변수 클래스. LeggedRobotCfg Class / LeggedRobotCfgPPO Class
│   │   ├── scripts
│   │   │   ├── train.py: 학습 실행 메인 코드. wandb 설정 셋팅. (Command-training 참고)
│   │   │   ├── play.py: 학습 완료 후 다양한 지형에서 여러 agent들의 보행 inference motion을 확인하는 코드.(Command-inference 참고)
│   │   │   └── mini_test.py:  학습 완료 후 다양한 지형에서 여러 agent들의 보행 inference motion을 확인하는 코드.(Command-inference 참고)
│   │   └── utils
│   │       ├── logger.py: play.py나 mini_test.py에서 사용되는 matplotlib plot을 위한 코드.
│   │       ├── task_registry.py: envs/__init__.py에 등록된 학습 환경 정보를 기반으로 환경과 알고리즘 연결 실행.
│   │       └── terrain.py: 보행하는 지형 클래스. LeggedRobot에서 참조.
│   │ 
│   └── resources/robots/a1: 로봇 플랫폼에 대한 정보.(urdf&mesh)
│
└── rsl_rl
    └── rsl_rl
        ├── algorithms
        │   └── ppo.py: PPO 알고리즘 코드. actor_critic.py의 Actor/Critic 클래스 사용.
        ├── modules
        │   └── actor_critic.py: Actor/Critic 클래스 코드. 
        ├── runners
        │   └── on_policy_runner.py: 강화학습 메인 loop(learn 함수)가 있는 OnPolicyRunner 클래스가 있는 파일. 
        │                            Base model은 OnPolicyRunner 클래스로, DreamWaQ model은 OnPolicyRunnerWaq 클래스로,
        │                            Estnet model은 OnPolicyRunnerEst 클래스로 학습 코드가 돌아감.
        │                            (강화학습 main loop 앞 단계[actor/critic network 이전 단계]의 변형 에따라 클래스 구분)
        ├── utils
        │   └── rms.py: CENet의 normal prior distribution 학습을 위한 Running Mean Std 클래스. 
        └── vae
            ├── cenet.py: Context-Aided Estimator Network(CENet) 클래스.
            └── estnet.py: 비교모델군인 Estimator 클래스.

```



## Result Graphs

약 1000 iteration 동안 학습 Reward Graph

![](./asset/two_models_rew.png)

### DreamWaQ model

- 학습 후 1개의 robot agent의 state plot
  - 1행: base state 중 x, y 방향의 속도와 yaw 방향의 command와 실제 측정 물리량 plot
  - 2행: CENet을 통한 예측된 estimated 속도와 실제 시뮬레이터에서 측정된 true 속도 plot
  - 3행: estimated 속도와 true 속도의 error plot
    - 1열: x, y, z 방향의 각 성분의 squared error
    - 2, 3열, x, y 방향의 mean squared error

![](./asset/a1_waq_est_vel.png)

### Base model

- 학습 후 1개의 robot agent의 state plot(DreamWaQ와 달리, estimated 속도가 없으므로 plot한 그래프가 다름.)
  - 1행: base state 중 x, y 방향의 속도와 yaw 방향의 command와 실제 측정 물리량 plot
  - 2행 1열/2열: 1개의 joint의 위치와 속도 
  - 2행 3열: base z 방향 속도
  - 3행 1열: 4개 발의 contact force
  - 3행 2/3열: 1개의 joint torque

![](./asset/a1_base_no_vel.png)

## Result Motions

### Walking Performance of a Reproduction Model in Different Terrains
- Smooth Slope / Rough Slope

![](./asset/1.gif)

- Stair Up / Stair Down

![](./asset/2.gif)

- Discrete / Mixed

![](./asset/3.gif)



### Comparative Analysis of Walking Motion Between the Reproduction Model and the Base Model

> small difference: naturalness of motion
> 
> big difference: foot stuck / unstable step

- Smooth Slope(small difference)

![](./asset/4.gif)

- Rough Slope(small difference)

![](./asset/5.gif)

- Stair Up(big difference)

![](./asset/6.gif)

- Stair Down(big difference)

![](./asset/7.gif)

- Discrete(big difference)

![](./asset/8.gif)


