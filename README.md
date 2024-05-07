# stonkdecision
testing decision transformer with stock trading

This repo contain the docker image for curating stock trading data from various source such as reinforcement learning trained agent; training a decision transformer model on those data; evlulating the model on a gym/gymnasium environment; and deploying the model.

Setup:
Simply spin up docker container using docker compose up at this root directory. It should create a link for accessing the jupyter server.

Demo:
- Curating ation-state-reward dataset
    - Open src/curate_demo.ipynb and run the cells inside. It should provide a step by step on how to curate dataset on interaction with the envrionment.
    - It show case the usage of helper functions on how to download stock price data, create gym environments, train stable-baseline RL-agent, sample RL-agent and traditional algorithem with the gym environment

- Train Decision transformer and evaluate its performance on gym environment
    - Open src/train_eval_decision_transformer.ipynb and run the cell inside.
    - It runs the program which train decision transformer on the curated dataset from last demo.
    - It also show case helper functions on evaluate the decision transformer in testing environment and visualise its action.
