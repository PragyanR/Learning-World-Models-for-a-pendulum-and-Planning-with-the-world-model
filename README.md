# Learning a world model for Pendulum + planning with it

```
pip install -r requirements.txt
python datacollection.py --episodes 100     
random transitions -> data/random_data.npz
python train.py                             
python evaluate.py --sac-steps 30000       
```

