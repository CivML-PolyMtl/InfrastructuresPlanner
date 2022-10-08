# Infrastructure Planner

---

Infrastructure Planner is a custom gym environment to plan interventions on transportation infrastrucutre.

## Getting Started

To get started, create a virtual environment in vscode

`
$ python -m venv .venv
`

Install the required packages by using the pip command in terminal:

`
$ pip install -r requirements.txt
`

Excute environment tests using `test_env.py`.

To perform analyses on vectorized environment:

`
$ pip install RayEnvWrapper
$ number_of_workers, envs_per_worker = 10, 2
$ base_env = infra_planner
$ env = WrapperRayVecEnv(base_env, number_of_workers, envs_per_worker)
$ env.step(0)
`
