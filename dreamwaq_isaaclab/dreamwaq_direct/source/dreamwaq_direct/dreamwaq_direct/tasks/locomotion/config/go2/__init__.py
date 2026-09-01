"""Go2 DreamWaQ DirectRLEnv configurations and Gymnasium registration."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="DreamWaQ-Direct-Go2-Base-v0",
    entry_point="dreamwaq_direct.tasks.locomotion.dreamwaq_env:DreamWaQEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2BaseDirectCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BasePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Direct-Go2-Base-Play-v0",
    entry_point="dreamwaq_direct.tasks.locomotion.dreamwaq_env:DreamWaQEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2BaseDirectCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BasePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Direct-Go2-Oracle-v0",
    entry_point="dreamwaq_direct.tasks.locomotion.dreamwaq_env:DreamWaQEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2OracleDirectCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2OraclePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Direct-Go2-Oracle-Play-v0",
    entry_point="dreamwaq_direct.tasks.locomotion.dreamwaq_env:DreamWaQEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2OracleDirectCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2OraclePPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Direct-Go2-Waq-v0",
    entry_point="dreamwaq_direct.tasks.locomotion.dreamwaq_env:DreamWaQEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2WaqDirectCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WaqPPORunnerCfg",
    },
)

gym.register(
    id="DreamWaQ-Direct-Go2-Waq-Play-v0",
    entry_point="dreamwaq_direct.tasks.locomotion.dreamwaq_env:DreamWaQEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2WaqDirectCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WaqPPORunnerCfg",
    },
)
