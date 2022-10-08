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
# Or,
python3 -m venv .venv
```

Activate the environment,

```{.py}
source .venv/bin/activate
# Or in windows,
".venv/scripts/activate.bat"
```

Install the required packages by using the pip command in terminal,

```{.py}
 pip install -r requirements.txt
```

Run environment tests using,

```{.py}
 python test_env.py
```

## Remarks

The Infra Planner package is originally developed based on the inspection and interventions database of the Transportation Ministry of Quebec (MTQ).

## Misc

To perform analyses on vectorized environment, a recommanded wrapper is (RayEnvWrapper):

```{ .py}
pip install RayEnvWrapper
number_of_workers, envs_per_worker = 10, 2
base_env = infra_planner
env = WrapperRayVecEnv(base_env, number_of_workers, envs_per_worker)
env.step(0)
```

  **Note: RayEnvWrapper is an external package and could be incompatible with the OS configureation.

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on the process for submitting pull requests.

## Authors

* **Zachary Hamida** - *Methodology, initial code and development* - [webpage](https://zachamida.github.io)
* **James-A. Goulet** - *Methodology* - [webpage](http://profs.polymtl.ca/jagoulet/Site/Goulet_web_page_MAIN.html)

## Acknowledgments

- The funding for this project is provided by the Transportation Ministry of Quebec Province (MTQ), Canada.
