#%%
from infra_planner import infra_planner

env = infra_planner(fixed_seed=1)
env.env.test_env(1000)


