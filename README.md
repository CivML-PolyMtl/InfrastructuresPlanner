# Infrastructure Planner

---

Infrastructure Planner is a custom gym environment to plan interventions on transportation infrastrucutre.

## Getting Started

To get started, create a virtual environment in vscode

```{.py}
 python -m venv .venv
```

Install the required packages by using the pip command in terminal:

```{.py}
 pip install -r requirements.txt
```

Excute environment tests using `test_env.py`.

To perform analyses on vectorized environment:

```{ .py}
pip install RayEnvWrapper
number_of_workers, envs_per_worker = 10, 2
base_env = infra_planner
env = WrapperRayVecEnv(base_env, number_of_workers, envs_per_worker)
env.step(0)
```

Note: RayEnvWrapper is an external package and could be incompatible with the OS configureation.
