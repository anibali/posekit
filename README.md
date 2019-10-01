# PoseKit

## Installing dependencies

A complete virtual environment with all dependencies for PoseKit can be setup using Conda and
Pipenv:

```bash
$ conda env create -f environment.yml
$ conda activate posekit
$ pipenv --python=$(conda run which python) --site-packages
```
