#%%
from infra_planner import infra_planner

# infra_lvl (0: element level, 1: category level, 2: bridge level )
env = infra_planner(fixed_seed=1, infra_lvl=2)

env.plotting = 1
env.test_env(episodes_num= 1)


