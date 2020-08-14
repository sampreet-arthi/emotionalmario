# Emotional-Mario

Test code for the Emotional Mario challenge of Media Eval 2020

## Toadstool Processing

[Toadstool](https://github.com/simula/toadstool) is a dataset of humans playing Super Mario Bros. The `process_toadstool.py` script can be used to replay the action from the dataset and generate frames which agents can be trained with. For example the following command will generate a dataset of frames from all runs in Toadstool. (This may take some time to execute)

```bash
python process_toadstool.py -i toadstool/toadstool/participants/ -o toadstool/processed/
```

For processing an individual `json` file of actions, the following format can be followed. Not that the output path must end in `participant_x` with `x` being the participat number.

```bash
python process_toadstool.py -i toadstool/toadstool/participants/participant_0/participant_0_session.json -o toadstool/processed/participant_0/
```

## Behavior Cloning

Behavior cloning involves training an agent on a datset of (observation, action) tuples taken from an expert. The `behavior_cloning.py` script can be used to train an agent on Toadstool. For example the following command will train the agent from data stores in a given directory for 100 epochs.

```bash
python behavior_cloning.py -i toadstool/processed/ -e 100
```
