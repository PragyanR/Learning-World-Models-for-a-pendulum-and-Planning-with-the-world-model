# Learning a world model for Pendulum + planning with it

```
pip install -r requirements.txt
python datacollection.py --episodes 100     # 20k random transitions -> data/random_data.npz
python train.py                             # -> checkpoints/worldmodel.pt
python evaluate.py --sac-steps 30000        # -> plots/*.png  (drop --sac-steps to skip SAC)
```

| file | role |
|---|---|
| `datacollection.py` | random-policy rollouts, saved per-episode |
| `worldmodel.py` | MLP (2 hidden layers, ReLU) predicting normalized delta of [cos, sin, thdot] |
| `train.py` | Adam, 90/10 split by episode, best-val checkpoint, multi-step error |
| `planner.py` | MPC with random shooting or CEM using the learned model |
| `evaluate.py` | random vs MPC vs SAC baseline, plots |
