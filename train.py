import argparse
import os
import logging

import torch
import torch.optim as optim

from utils.misc import Params, set_logger
import dataloader.data_loader as data_loader
import model.my_net.my_net as net
from utils.loss import loss_fn
from utils.metrics import metrics
from engine import train_and_evaluate


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data/64x64_SIGNS",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--model_dir",
        default="experiments/base_model",
        help="Directory containing params.json",
    )
    parser.add_argument(
        "--restore_file",
        default=None,
        help="Optional, name of the file in --model_dir containing weights to reload before training",
    )

    return parser


if __name__ == "__main__":
    args = get_parse().parse_args()
    json_path = os.path.join(args.model_dir, "params.json")

    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"

    params = Params(json_path)
    params.cuda = torch.cuda.is_available()

    torch.manual_seed(42)
    if params.cuda:
        torch.cuda.manual_seed(42)

    set_logger(os.path.join(args.model_dir, "train.log"))

    logging.info("Loading the datasets...")

    dataloaders = data_loader.fetch_dataloader(["train", "val"], args.data_dir, params)
    train_dl = dataloaders["train"]
    val_dl = dataloaders["val"]

    logging.info("- done. ")

    model = net.MyNet(params).cuda() if params.cuda else net.MyNet(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    loss_fn = loss_fn
    metrics = metrics

    logging.info(f"Starting training for {params.num_epochs} epochs(s)")
    train_and_evaluate(
        model,
        train_dl,
        val_dl,
        optimizer,
        loss_fn,
        metrics,
        params,
        args.model_dir,
        args.restore_file,
    )
