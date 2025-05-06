import torch
import numpy as np

def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)  

def MSE(y_true, y_pred, dataset_name):
    if 'pems' in dataset_name or 'metr' in dataset_name or 'air' in dataset_name:
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
    else:
        mse = np.square(y_pred - y_true)
        mse = np.mean(mse)
    return mse

def RMSE(y_true, y_pred, dataset_name):
    if 'pems' in dataset_name or 'metr' in dataset_name or 'air' in dataset_name or 'aqi' in dataset_name:
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
    else:
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean(rmse))
    return rmse

def MAE(y_true, y_pred, dataset_name):
    if 'pems' in dataset_name or 'metr' in dataset_name or 'air' in dataset_name or 'aqi' in dataset_name:
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
    else:
        mae = np.abs(y_pred - y_true)
        mae = np.mean(mae)
    return mae

def MAPE(y_true, y_pred, dataset_name, null_val=0):
    epsilon = 1e-10
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = np.not_equal(y_true, null_val)
    mask = mask.astype("float32")
    mask /= np.mean(mask)
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true_safe))
    mape = np.nan_to_num(mask * mape)
    return np.mean(mape) * 100

def cal_metrics(y_true, y_pred, dataset_name):
    return (
        RMSE(y_true, y_pred, dataset_name),
        MAE(y_true, y_pred, dataset_name),
        MAPE(y_true, y_pred, dataset_name),
    )
