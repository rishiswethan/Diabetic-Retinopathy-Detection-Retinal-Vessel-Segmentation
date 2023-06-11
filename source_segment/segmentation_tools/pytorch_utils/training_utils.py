from torch.optim import Adam as adam_opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import gc
import traceback
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
import torchmetrics


device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=RuntimeWarning, module='tkinter')


def _evaluate(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device=device,
        custom_metrics=None,
        custom_metrics_args_dict=None
):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in val_loader:
            # Move the input tensors to the GPU if available
            batch = [tensor.to(device) for tensor in batch]
            outputs.append(model.validation_step(batch, custom_metrics=custom_metrics, args_dict=custom_metrics_args_dict))

    # return the custom metrics only if they were provided
    if custom_metrics is not None:
        # print("outputs: ", outputs)
        cust_metric_vals = []
        for i in range(len(outputs)):
            cust_metrics = []
            for j in range(len(custom_metrics)):
                cust_metrics.append(outputs[i]["val_" + custom_metrics[j]].cpu().numpy())
            cust_metric_vals.append(cust_metrics)

        return cust_metric_vals

    return model.validation_epoch_end(outputs)


def _accuracy(args_dict=None):
    def accuracy(outputs: torch.Tensor, labels: torch.Tensor):
        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)

        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    return accuracy


def _f1_score(args_dict=None):
    num_classes = args_dict["num_classes"]
    def f1_score(outputs: torch.Tensor, labels: torch.Tensor):

        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)

        f1 = torchmetrics.F1Score(num_classes=num_classes, average='weighted', task='multiclass').to(device)
        f1 = f1(preds, labels)

        return f1
    return f1_score

def _precision_score(args_dict=None):
    num_classes = args_dict["num_classes"]
    def precision_score(outputs: torch.Tensor, labels: torch.Tensor):

        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)

        precision = torchmetrics.Precision(num_classes=num_classes, average='weighted', task='multiclass').to(device)
        precision = precision(preds, labels)

        return precision
    return precision_score


def _recall_score(args_dict=None):
    num_classes = args_dict["num_classes"]
    def recall_score(outputs: torch.Tensor, labels: torch.Tensor):

        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)

        recall = torchmetrics.Recall(num_classes=num_classes, average='weighted', task='multiclass').to(device)
        recall = recall(preds, labels)

        return recall
    return recall_score


def _quadratic_weighted_kappa(args_dict=None):
    def quadratic_weighted_kappa(outputs: torch.Tensor, labels: torch.Tensor):
        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)

        # Convert tensors to numpy arrays for use with scikit-learn
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            kappa = cohen_kappa_score(labels_np, preds_np, weights='quadratic')

        if np.isnan(kappa) or np.isinf(kappa):
            kappa = 0

        return torch.tensor(kappa)

    return quadratic_weighted_kappa


def _confusion_matrix_elements_multiclass(args_dict):
    num_classes = args_dict["num_classes"]

    def confusion_matrix_elements_multiclass(
            outputs: torch.Tensor,
            labels: torch.Tensor,
    ):
        preds = torch.argmax(outputs, dim=1)
        labels = torch.argmax(labels, dim=1)

        cm = torch.zeros((num_classes, num_classes), device=outputs.device)

        for i in range(labels.shape[0]):
            cm[labels[i], preds[i]] += 1

        return cm

    return confusion_matrix_elements_multiclass


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, **kwargs):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()


def get_metric(metric_name="accuracy", args_dict=None):
    """
    Returns the metric function given the metric name
    """
    metrics = {
        "accuracy": _accuracy,
        "f1_score": _f1_score,
        "quadratic_weighted_kappa": _quadratic_weighted_kappa,
        "precision_score": _precision_score,
        "recall_score": _recall_score,
        "confusion_matrix_elements_multiclass": _confusion_matrix_elements_multiclass,
    }

    return metrics[metric_name](args_dict=args_dict)


def _calculate_other_metrics(outputs: torch.Tensor, labels: torch.Tensor, other_metrics, args_dict=None):

    other_metrics_values = {}
    for metric in other_metrics:
        metric_func = get_metric(metric, args_dict)
        other_metrics_values[metric] = metric_func(outputs, labels)

    return other_metrics_values


def fit(
        epochs: int,
        lr: float,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        weight_decay: float=None,
        callbacks_function=None,
        continue_training=False,
        opt_func=adam_opt,
        device=device,
        num_retries_inner=10,
        max_retry=10,
        evaluate=_evaluate,
):
    """
    Meant to resemble the fit function in keras.

    Parameters
    ----------
    epochs - Set this to a high number and use callbacks to stop training early
    lr - Initial learning rate in case of a scheduler
    weight_decay - Weight decay to be fed to the optimizer
    model - The model to train. Must inherit from CustomModelBase of this module
    train_loader - The training data loader
    val_loader - The validation data loader
    callbacks_function - A function that takes the model and returns a list of callbacks
    opt_func - The optimizer function to use
    device - The device to use
    num_retries_inner - Number of times to retry training if it fails
    max_retry - Maximum number of times to retry training if anything other than training_step fails, like dataloader
    evaluate - The function to use for evaluation, defaults to _evaluate()

    Returns
    -------
    history - A list of dictionaries containing the loss and accuracy for each epoch

    """

    if weight_decay is not None:
        optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)
    else:
        optimizer = opt_func(model.parameters(), lr)

    model.to(device)
    defined_callbacks = None  # must be None so that it can be defined in the function when it is called for the first time
    num_retry = 0
    history = []

    for epoch in range(epochs):
        model.train()  # Make sure the model is in training mode at each epoch, because it is set to eval() in evaluate()
        train_losses = []
        accuracies = []
        other_metrics_values = []
        print("LR: ", optimizer.param_groups[0]['lr'])
        # Wrap the train_loader with tqdm to create a progress bar

        while num_retry < max_retry:
            try:
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", delay=1)

                for batch in progress_bar:
                    batch = [tensor.to(device) for tensor in batch]

                    # run the training step many times until it works
                    flag = False
                    for i in range(num_retries_inner):
                        try:
                            loss, acc, oth_metrics_value = model.training_step(batch)
                            flag = True
                            break
                        except:
                            if i == num_retries_inner - 1:
                                traceback.print_exc()

                            # try cleaning the cache
                            torch.cuda.empty_cache()
                            gc.collect()

                    if not flag:
                        raise RuntimeError(f"Training step failed {num_retries_inner} times")

                    train_losses.append(loss)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    accuracies.append(acc)
                    # Update the progress bar with the current loss and accuracy
                    progress_bar.set_postfix(loss=loss.item(), accuracy=acc.item())

                    if oth_metrics_value is not None:
                        other_metrics_values.append(oth_metrics_value)

                num_retry = 0
                break
            except:
                # try cleaning the cache
                torch.cuda.empty_cache()
                gc.collect()

                num_retry += 1
                if num_retry < max_retry:
                    continue
                else:
                    traceback.print_exc()
                    raise RuntimeError(f"Training failed {max_retry} times")

        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).cpu().mean().item()
        result['train_acc'] = torch.stack(accuracies).cpu().mean().item()

        # add other metrics to result
        for metric_dict in other_metrics_values:
            for metric, value in metric_dict.items():
                result["train_" + metric] = value.cpu().item()

        if callbacks_function is not None:
            defined_callbacks, stop_flag = callbacks_function(
                optimiser=optimizer,
                result=result,
                model=model,
                defined_callbacks=defined_callbacks,
                continue_training=continue_training,
            )

            model.epoch_end(epoch, result)
            history.append(result)

            if stop_flag:
                print("Early stopping triggered")
                break

    return history


class CustomModelBase(torch.nn.Module):
    """
    Base class for custom models. This class is meant to be inherited from and not used directly. Override the training_step, and validation_step if you want to use custom loss functions.
    This class must be inherited in case you want to use the fit() function defined in this module.

    Parameters
    ----------
    class_weights : torch.Tensor
        The class weights to use for the loss function. This should be a 1D tensor with the same number of elements as the number of classes.
        Ideally, they should be normalized so that the sum of the weights is 1.
        Examples:
        - [0.11765947096395296, 0.21896579990935885, 0.2190948310230356, 0.23457661088081475, 0.2097032872228378]
    loss_function
        The loss function to use. Must be touch.nn.functional. This should be a function that takes in the model outputs, the labels, and any other arguments that are needed.
        Defaults to torch.nn.functional.cross_entropy.
    acc_func_name
        The accuracy function to use. This should be a string that is a key in the dictionary defined in get_metrics(). Defaults to "accuracy".
    other_acc_metrics
        A list of other metrics to calculate. This should be a list of strings that are keys in the dictionary defined in get_metrics(). Defaults to ["f1_score"].
    """

    def __init__(
            self,
            class_weights=None,
            loss_function=F.cross_entropy,
            acc_func_name="accuracy",
            other_acc_metrics=["f1_score"],
            num_classes=5,
    ):
        super(CustomModelBase, self).__init__()
        self.class_weights = class_weights
        self.loss_function = loss_function
        self.args_dict = {"num_classes": num_classes}
        self.accuracy_function = get_metric(acc_func_name, args_dict=self.args_dict)
        self.other_metrics_function = _calculate_other_metrics
        self.other_metrics = other_acc_metrics
        self.num_classes = num_classes

    def training_step(self, batch: list):
        """
        The training step. This is meant to be overridden if you want to use a custom loss function.
        Parameters
        ----------
        batch : list of torch.Tensor
            Examples:
            - batch = [tensor.to(device) for tensor in batch]

        Returns
        -------
        loss : torch.Tensor
        acc : torch.Tensor
        """

        images, labels = batch
        out = self(images)  # Generate predictions

        loss = self.loss_function(out, labels, weight=self.class_weights)  # Calculate loss with class weights
        acc = self.accuracy_function(out, labels)  # Calculate accuracy

        other_metrics = None
        if self.other_metrics is not None:
            other_metrics = self.other_metrics_function(out, labels, self.other_metrics, self.args_dict)

        return loss, acc, other_metrics

    def validation_step(self, batch: list, custom_metrics=None, args_dict=None):
        """
        The validation step. This is meant to be overridden if you want to use a custom loss function.
        Parameters
        ----------
        batch : list of torch.Tensor
            Examples:
            - batch = [tensor.to(device) for tensor in batch]

        Returns
        -------
        loss : torch.Tensor
        acc : torch.Tensor
        """

        if custom_metrics is None:
            custom_metrics = self.other_metrics

        if args_dict is None:
            args_dict = self.args_dict

        images, labels = batch
        out = self(images)  # Generate predictions

        loss = self.loss_function(out, labels, weight=self.class_weights)  # Calculate loss with class weights
        acc = self.accuracy_function(out, labels)  # Calculate accuracy

        other_metrics = None
        if custom_metrics is not None:
            other_metrics = self.other_metrics_function(out, labels, custom_metrics, args_dict=args_dict)

        ret_dict = {f"val_loss": loss, f"val_acc": acc}
        for metric in custom_metrics:
            ret_dict['val_' + metric] = other_metrics[metric]

        return ret_dict

    def validation_epoch_end(self, outputs):
        """
        Used to combine the results in the validation step and return the average loss and accuracy. Override this if you want to use custom metrics.
        """

        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies

        other_metrics = None
        if self.other_metrics is not None:
            other_metrics = {}
            for metric in self.other_metrics:
                batch_accs = [x['val_' + metric] for x in outputs]
                other_metrics[metric] = torch.stack(batch_accs).mean()

        ret_dict = {"val_loss": epoch_loss.cpu().item(), "val_acc": epoch_acc.cpu().item()}
        for metric in self.other_metrics:
            ret_dict['val_' + metric] = other_metrics[metric].cpu().item()

        return ret_dict

    def epoch_end(self, epoch, result):
        """
        Used to print the results of the epoch. Called at the end of each epoch in fit()
        """

        print(
            f"train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}\n"
            f"train_acc: {result['train_acc']:.4f}, val_acc: {result['val_acc']:.4f}"
        )
        for metric in self.other_metrics:
            print(f"train_{metric}: {result['train_' + metric]:.4f}, val_{metric}: {result['val_' + metric]:.4f}")

        print()
