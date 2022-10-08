# Infrastructure Planner

---

Infrastructure Planner is a custom gym environment to plan interventions on transportation infrastrucutre.

## Prerequisites

- Python 3.

- Pytorch: load pre-traind models (Optional).

- Access to GPU computing (Optional)

## Getting Started

To get started, clone the repo,

```{.py}
git clone https://github.com/CivML-PolyMtl/InfrastructuresPlanner.git
```

Access the project folder,

```{.py}
cd InfrastructuresPlanner
```

Create a virtual environment,

```{.py}
python -m venv .venv
```

Or,

```{.py}
python3 -m venv .venv
```

Activate the environment,

```{.py}
source .venv/bin/activate
```

Install the required packages by using the pip command in terminal,

```{.py}
 pip install -r requirements.txt
```

Run environment tests using,

```{.py}
 python test_env.py
```

To perform analyses on vectorized environment:

```{ .py}
pip install RayEnvWrapper
number_of_workers, envs_per_worker = 10, 2
base_env = infra_planner
env = WrapperRayVecEnv(base_env, number_of_workers, envs_per_worker)
env.step(0)
```

Note: RayEnvWrapper is an external package and could be incompatible with the OS configureation.
