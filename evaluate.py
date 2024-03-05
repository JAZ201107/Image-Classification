import argparse
import logging
import os


import numpy as np
import torch
from torch.autograd import Variable
from utils.misc import Params, set_logger, load_checkpoint, save_dict_to_json
from model.my_net import my_net as net
from dataloader import data_loader
from engine import evaluate
from utils.loss import loss_fn
from utils.metrics import metrics


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
        default="best",
        help="name of the file in --model_dir \
                        containing weights to load",
    )

    return parser


if __name__ == "__main__":
    args = get_parse().parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    params = Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    set_logger(os.path.join(args.model_dir, "evaluate.log"))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(["test"], args.data_dir, params)
    test_dl = dataloaders["test"]

    logging.info("- done.")

    # Define the model
    model = net.MyNet(params).cuda() if params.cuda else net.MyNet(params)

    loss_fn = loss_fn
    metrics = metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    load_checkpoint(os.path.join(args.model_dir, args.restore_file + ".pth.tar"), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file)
    )
    save_dict_to_json(test_metrics, save_path)
