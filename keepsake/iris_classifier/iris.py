import argparse
import keepsake
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.autograd import Variable

from configobj import ConfigObj
from validate import Validator

import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="iris classifier")

    parser.add_argument("config_path", type=str)

    args = parser.parse_args()
    return args


def train(config_path):

    cfgspec = ConfigObj("config/configspec.ini", list_values=False, encoding='UTF8', _inspec=True)
    cfg = ConfigObj(config_path, configspec=cfgspec)
    cfg.validate(Validator())

    subprocess.Popen(['cp', args.config_path, 'experiment_dir/config.ini']).wait()

    ########################################################################
    # Create an "experiment". This represents a run of your training script.
    # It saves your parameters and any file(s) you specify.

    # path -> path to upload on a per experiment basis
    # params -> dictionary of parameters to track per experiment
    experiment = keepsake.init(
        path="experiment_dir/",
        params=cfg,
    )
    ########################################################################

    print("Downloading data set...")
    iris = load_iris()
    train_features, val_features, train_labels, val_labels = train_test_split(
        iris.data,
        iris.target,
        train_size=cfg["train_size"],
        test_size=1 - cfg["train_size"],
        random_state=0,
        stratify=iris.target,
    )
    train_features = torch.FloatTensor(train_features)
    val_features = torch.FloatTensor(val_features)
    train_labels = torch.LongTensor(train_labels)
    val_labels = torch.LongTensor(val_labels)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, cfg["middle_dim"]), nn.ReLU(), nn.Linear(cfg["middle_dim"], 3),)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(val_features)
            acc = (output.argmax(1) == val_labels).float().sum() / len(val_labels)

        print(
            "Epoch {}, train loss: {:.3f}, validation accuracy: {:.3f}".format(
                epoch, loss.item(), acc
            )
        )
        torch.save(model, "checkpoint_dir/model.pth")

        #############################################################################
        # Create a checkpoint within the experiment.
        # This saves the metrics at that point, and makes a copy of the file
        # or directory given, which could be weights and/ot any other artifacts.

        # path -> path to upload on a per checkpoint basis
        # step -> var used to track checkpoint's step number (optional)
        # metrics -> dictionary of metrics to track per checkpoint
        # primary_metric -> metric used by keepsake to compare checkpoint performance
        experiment.checkpoint(
            path="checkpoint_dir/",
            step=epoch,
            metrics={"loss": loss.item(), "accuracy": acc},
            primary_metric=("loss", "minimize"),
        )
        #############################################################################


if __name__ == "__main__":
    args = parse_args()
    train(args.config_path)