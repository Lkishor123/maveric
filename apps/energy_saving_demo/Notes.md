1. Create a energy_saving_demo_app.py:
	- Do the BDT training as per CCO example app
	- Load the BDT model
	- Call energy saving class and try to make sense of flow there.

2. understading:
	- Once BDT is trained, we split our traffic demand data in train and test (24 tick: 17:7)
	- Train the RL on 17 ticks, save the RL model and then run inference/simulation on rest 7ticks 