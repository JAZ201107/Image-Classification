import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import logging

import config
from utils.misc import (
    load_checkpoint,
    save_checkpoint,
    save_dict_to_json,
    RunningAverage,
)


def evaluate(model, loss_fn, dataloader, metrics, params):

    model.eval()
    summ = []

    with tqdm(total=len(dataloader)) as t:
        for data_batch, labels_batch in dataloader:
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(
                    non_blocking=True
                ), labels_batch.cuda(non_blocking=True)

            # data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()

            summary_batch = {
                metric: metrics[metric](output_batch, labels_batch)
                for metric in metrics
            }

            summary_batch["loss"] = loss.item()
            summ.append(summary_batch)

            t.set_postfix(loss="{:05.3f}".format(loss.item()))
            t.update()

    # compute all mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    # metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )

    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def train_one_epoch(
    model: nn.Module, optimizer: torch.optim, loss_fn, dataloader, metrics, params
):

    model.train()

    summ = []
    loss_avg = RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True
                ), labels_batch.cuda(non_blocking=True)

            # train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {
                    metric: metrics[metric](output_batch, labels_batch)
                    for metric in metrics
                }
                summary_batch["loss"] = loss.item()
                summ.append(summary_batch)

            loss_avg.update(loss.item())

            t.set_postfix(loss="{:05.3f}".format(loss_avg()))
            t.update()

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " : ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    metrics,
    params,
    model_dir,
    restore_file=None,
):
    if restore_file is not None:
        restore_path = os.path.join(model, restore_file, config.MODEL_END)

        logging.info(f"Restoring parameters from {restore_path}")
        load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch + 1} / {params.num_epochs}")

        train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataloader=train_dataloader,
            metrics=metrics,
            params=params,
        )

        val_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            dataloader=val_dataloader,
            metrics=metrics,
            params=params,
        )

        val_acc = val_metrics["accuracy"]
        is_best = val_acc >= best_val_acc

        # Save weights for best
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optim_dic": optimizer.state_dict(),
            },
            is_best=is_best,
            checkpoint=model_dir,
        )

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # save best val metrics in a json file
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")

            save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)
