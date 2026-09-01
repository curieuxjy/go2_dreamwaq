"""Go2 DreamWaQ environment configurations and Gymnasium registration.

Lecture scope: **PPO only**, three observation variants — Base / Oracle / Waq.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="DreamWaQ-Manager-Go2-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_base_cfg:Go2BaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BasePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Base-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_base_cfg:Go2BaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BasePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Oracle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_oracle_cfg:Go2OracleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2OraclePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Oracle-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_oracle_cfg:Go2OracleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2OraclePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Waq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_waq_cfg:Go2WaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WaqPPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Manager-Go2-Waq-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_waq_cfg:Go2WaqEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WaqPPORunnerCfg",
    },
)

##
# Stage 1 강의의 실제 비교축 — 공식 env 레시피 위에서 actor 관측만 수술한 3변형.
#
# 원본 DreamWaQ 레시피(±1 m/s x 3축 push 매 1초, DreamWaQ 보상 세트)로는 모든 알고리즘이
# underlying 0.16~0.26 에서 정체하고 보행에 실패한다. 원인이 CENet 이 아니라 env 설계임을
# 확인해 전 매트릭스를 공식 레시피로 통일했다 (report.qmd §1).
#
#   Base   actor 45 (blind)                        하한선
#   Oracle actor 48 (+ true lin_vel)               속도 정보 상한선
#   Waq    actor 45 -> 64 (+ CENet est_vel+context) DreamWaQ 제안
#
# critic 은 셋 다 privileged (공식 관측 전체 + rough 는 height_scan).
##

# Flat Oracle used to point at the stock UnitreeGo2FlatEnvCfg; it now needs a local subclass
# so that the DreamWaQ Table-I rewards apply to this arm too (PAPER.md §2).
for _terrain, _envcls in (("Flat", "OracleOfficialFlatEnvCfg"), ("Rough", "OracleOfficialRoughEnvCfg")):
    for _suffix, _envsfx in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"DreamWaQ-OracleDwq-{_terrain}-PPO{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.go2_waq_official_cfg:{_envcls}{_envsfx}",
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_ppo_cfg:Go2OracleDwqHparams{_terrain}PPORunnerCfg"
                ),
            },
        )

for _terrain, _envcls in (("Flat", "WaqOfficialFlatEnvCfg"), ("Rough", "WaqOfficialRoughEnvCfg")):
    for _suffix, _envsfx in (("", ""), ("-Play", "_PLAY")):
        # Base: blind actor 45 + privileged critic, Waq 와 같은 hparam (Waq - Base = CENet 순수 기여)
        gym.register(
            id=f"DreamWaQ-BaseDwq-{_terrain}-PPO{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.go2_waq_official_cfg:{_envcls}{_envsfx}",
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BaseDwqHparams{_terrain}PPORunnerCfg"
                ),
            },
        )
        # Waq: 같은 env, CENet 러너
        gym.register(
            id=f"DreamWaQ-Waq-Official-{_terrain}-PPO{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.go2_waq_official_cfg:{_envcls}{_envsfx}",
                "rsl_rl_cfg_entry_point": (
                    f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WaqOfficial{_terrain}PPORunnerCfg"
                ),
            },
        )


##
# 논문이 실제로 말한 oracle — 지형 높이맵을 보는 정책 (rough 전용).
#
# 논문 Fig.3 캡션: "The oracle policy has access to the height map measurement of the robot's
# surroundings", §III-B: "despite walking without exteroception, DreamWaQ performs almost as well
# as the oracle policy that has direct access to the surrounding terrain's height map."
#
# 위의 OracleDwq 는 *속도* 오라클이라 논문이 아무 주장도 하지 않은 축이다. 이 arm 이 있어야
# "눈먼 Waq 가 지형을 보는 정책에 근접하는가" 라는 논문의 주장을 시험할 수 있다.
# flat 은 지형 정보가 없으므로 rough 만 등록한다.
##
for _suffix, _envsfx in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"DreamWaQ-TerrainOracle-Rough-PPO{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.go2_waq_official_cfg:TerrainOracleOfficialRoughEnvCfg{_envsfx}"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:Go2TerrainOracleRoughPPORunnerCfg"
            ),
        },
    )
