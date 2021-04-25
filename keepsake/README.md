# Keepsake

## iris classifier

This example runs an iris classifier neural network model and logs it using keepsake.

To run the code, navigate to the iris_classifier directory and run the following command:

`python iris.py config/default.ini`

### How the code works:

1. Takes the config file you specify, loads it, and copies it to the `experiment_dir/` directory.

2. Creates a new keepsake experiment and saves it in the `repo/` directory. This saves the contents of `experiment_dir/` as well as the parameters associated with our experiment

3. Loads the iris dataset, defines a neural network architecture, and trains the model.

4. Saves a model checkpoint and the associated performance metrics at each epoch.