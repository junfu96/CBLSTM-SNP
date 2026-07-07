import numpy as np
import pandas as pd
import os
import tensorflow as tf

tf.config.run_functions_eagerly(True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN, LSTM, GRU, Conv1D

from recurrent import LSTMSNPCell
import matplotlib.pyplot as plt
TIME_STEPS = 5
EPOCHS = 100

DATASETS = ["sp500", "milk", "closings", "lake", "MG"]
MODES = ["lstm", "4gate", "5gate", "5gate_lambda"]

# ========= 数据加载 =========
def load_data(dataset_name):
    if dataset_name == "sp500":
        df = pd.read_csv("./data/sp500.csv")
        return df["Open"].values.astype("float32")

    elif dataset_name == "milk":
        df = pd.read_csv("./data/milk.csv")
        return df.iloc[:,1].values.astype("float32")

    elif dataset_name == "closings":
        df = pd.read_csv("./data/closings.csv")
        return df.iloc[:,1].values.astype("float32")

    elif dataset_name == "lake":
        df = pd.read_csv("./data/lake.csv")
        return df.iloc[:,1].values.astype("float32")

    elif dataset_name == "MG":
        df = pd.read_csv("./data/MG.csv", comment='#')
        return df["value"].values.astype("float32")


def difference(data):
    return np.diff(data)


def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)


def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(train.reshape(-1,1))
    return scaler, scaler.transform(train.reshape(-1,1)), scaler.transform(test.reshape(-1,1))


def invert_scale(scaler, y):
    return scaler.inverse_transform([[y]])[0,0]


def build_model(mode):

    model = Sequential()

    # ===== LSTM =====
    if mode == "lstm":

        model.add(
            LSTM(
                8,
                input_shape=(TIME_STEPS,1)
            )
        )

    # ===== 4gate =====
    elif mode == "4gate":

        model.add(
            RNN(
                LSTMSNPCell(
                    8,
                    use_compensation=False
                ),
                input_shape=(TIME_STEPS,1)
            )
        )

    # ===== 5gate =====
    elif mode == "5gate":

        model.add(
            RNN(
                LSTMSNPCell(
                    8,
                    use_compensation=True,
                    lambda_c=1.0
                ),
                input_shape=(TIME_STEPS,1)
            )
        )

    # ===== 5gate_lambda =====
    elif mode == "5gate_lambda":

        model.add(
            RNN(
                LSTMSNPCell(
                    8,
                    use_compensation=True,
                    lambda_c=0.5
                ),
                input_shape=(TIME_STEPS,1)
            )
        )

    model.add(Dense(1))

    model.compile(
        loss='mse',
        optimizer='adam'
    )

    return model


class EpochPredictionCallback(tf.keras.callbacks.Callback):

    def __init__(self, X_test, scaler, save_path,
                 state_path=None,
                 comp_path=None):

        self.X_test = X_test
        self.scaler = scaler
        self.save_path = save_path

        self.state_path = state_path
        self.comp_path = comp_path

        self.epoch_preds = []

        self.state_history = []
        self.comp_history = []

    def on_epoch_end(self, epoch, logs=None):

        preds = self.model.predict(self.X_test, verbose=0)

        preds = np.array([
            invert_scale(self.scaler, p[0])
            for p in preds
        ])

        self.epoch_preds.append(preds)

        # ===== 获取 RNN cell =====
        cell = self.model.layers[0].cell

        # ===== 保存 state =====
        if hasattr(cell, "last_state_norm"):

            self.state_history.append(
                float(cell.last_state_norm.numpy())
            )

        # ===== 保存 compensation =====
        if hasattr(cell, "last_comp_norm"):

            self.comp_history.append(
                float(cell.last_comp_norm.numpy())
            )

    def save(self):

        np.save(
            self.save_path,
            np.array(self.epoch_preds)
        )

        if self.state_path is not None:

            np.save(
                self.state_path,
                np.array(self.state_history)
            )

        if self.comp_path is not None:

            np.save(
                self.comp_path,
                np.array(self.comp_history)
            )


# ========= 单次运行 =========
def run_once(dataset, mode):

    raw = load_data(dataset)
    diff = difference(raw)

    X, y = create_dataset(diff, TIME_STEPS)

    split = int(len(X)*0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler, X_train, X_test = scale(X_train.flatten(), X_test.flatten())
    _, y_train, y_test = scale(y_train, y_test)

    X_train = X_train.reshape((-1, TIME_STEPS, 1))
    X_test = X_test.reshape((-1, TIME_STEPS, 1))

    model = build_model(mode)

    os.makedirs(f"./results/{dataset}", exist_ok=True)

    callback = EpochPredictionCallback(

        X_test,
        scaler,

        f"./results/{dataset}/{mode}_epoch.npy",

        state_path=f"./results/{dataset}/{mode}_state.npy",

        comp_path=f"./results/{dataset}/{mode}_comp.npy"
    )

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=1,
        verbose=0,
        callbacks=[callback]
    )

    callback.save()

    preds = model.predict(X_test, verbose=0)
    preds = np.array([invert_scale(scaler, p[0]) for p in preds])
    y_test = np.array([invert_scale(scaler, t[0]) for t in y_test])

    return preds, y_test


# ========= 主入口 =========
if __name__ == "__main__":
    for dataset in DATASETS:
        print(f"\n===== DATASET: {dataset} =====")

        for mode in MODES:
            preds, gt = run_once(dataset, mode)

            np.savetxt(f"./results/{dataset}/{mode}.csv", preds, delimiter=',')
            np.savetxt(f"./results/{dataset}/gt.csv", gt, delimiter=',')

            print(f"{mode} done.")
            
            
                    # ===== compare state evolution =====

            state_4gate = np.load(
                f"./results/{dataset}/4gate_state.npy"
            )

            state_5gate = np.load(
                f"./results/{dataset}/5gate_lambda_state.npy"
            )

            plt.figure(figsize=(8,4))

            plt.plot(
                state_4gate,
                label="LSTM-SNP (4-gate)"
            )

            plt.plot(
                state_5gate,
                label="CBLSTM-SNP (5-gate)"
            )

            plt.xlabel("Epoch")

            plt.ylabel("State magnitude")

            plt.title("Comparison of state evolution")

            plt.legend()

            plt.tight_layout()

            plt.savefig(
                f"./results/{dataset}/state_compare.png",
                dpi=300
            )

            plt.close()
            
            
            # 画第二张图
            comp_data = np.load(
                f"./results/{dataset}/5gate_lambda_comp.npy"
            )

            plt.figure(figsize=(8,4))

            plt.plot(comp_data)

            plt.xlabel("Epoch")

            plt.ylabel("Compensation magnitude")

            plt.title("Compensation dynamics")

            plt.tight_layout()

            plt.savefig(
                f"./results/{dataset}/compensation.png",
                dpi=300
            )

            plt.close()
            
            # # ====================图像绘制-单张图像3张=======================
            #     # ===== state evolution =====
            # if mode in ["4gate", "5gate_lambda"]:

            #     state_data = np.load(
            #         f"./results/{dataset}/{mode}_state.npy"
            #     )

            #     plt.figure(figsize=(8,4))
            #     plt.plot(state_data)

            #     plt.xlabel("Time step")
            #     plt.ylabel("Hidden state magnitude")

            #     plt.title(f"State evolution ({mode})")

            #     plt.tight_layout()

            #     plt.savefig(
            #         f"./results/{dataset}/{mode}_state.png",
            #         dpi=300
            #     )

            #     plt.close()
            # # ===== compensation dynamics =====
            # if mode == "5gate_lambda":

            #     comp_data = np.load(
            #         f"./results/{dataset}/{mode}_comp.npy"
            #     )

            #     plt.figure(figsize=(8,4))
            #     plt.plot(comp_data)

            #     plt.xlabel("Time step")
            #     plt.ylabel("Compensation magnitude")

            #     plt.title("Compensation dynamics")

            #     plt.tight_layout()

            #     plt.savefig(
            #         f"./results/{dataset}/{mode}_comp.png",
            #         dpi=300
            #     )

            #     plt.close()


        
        
        
    
# #----------------------------------------------RMSE-----------------------------------------
# # =========================
# # 模型构建
# # =========================
# def build_model(model_name):

#     model = Sequential()

#     # ===== LSTM =====
#     if model_name == "LSTM":
#         model.add(LSTM(8, input_shape=(TIME_STEPS,1)))

#     # ===== GRU =====
#     elif model_name == "GRU":
#         model.add(GRU(8, input_shape=(TIME_STEPS,1)))

#     # ===== LSTM-SNP (4gate) =====
#     elif model_name == "LSTM-SNP":
#         model.add(RNN(LSTMSNPCell(8, use_compensation=False),
#                       input_shape=(TIME_STEPS,1)))

#     # ===== CBLSTM-SNP (Conv + 5gate_lambda) =====
#     elif model_name == "CBLSTM-SNP":
#         model.add(Conv1D(filters=8, kernel_size=2,
#                          activation='relu',
#                          input_shape=(TIME_STEPS,1)))

#         model.add(RNN(LSTMSNPCell(8, use_compensation=True, lambda_c=0.5)))

#     model.add(Dense(1))
#     model.compile(loss='mse', optimizer='adam')

#     return model


# # =========================
# # 单次运行
# # =========================
# def run_once(dataset, model_name):

#     raw = load_data(dataset)
#     diff = difference(raw)

#     X, y = create_dataset(diff, TIME_STEPS)

#     split = int(len(X)*0.7)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]

#     scaler, X_train, X_test = scale(X_train.flatten(), X_test.flatten())
#     _, y_train, y_test = scale(y_train, y_test)

#     X_train = X_train.reshape((-1, TIME_STEPS, 1))
#     X_test = X_test.reshape((-1, TIME_STEPS, 1))

#     model = build_model(model_name)

#     model.fit(X_train, y_train,
#               epochs=EPOCHS,
#               batch_size=1,
#               verbose=0)

#     preds = model.predict(X_test, verbose=0)

#     preds = np.array([invert_scale(scaler, p[0]) for p in preds])
#     y_test = np.array([invert_scale(scaler, t[0]) for t in y_test])

#     rmse = np.sqrt(mean_squared_error(y_test, preds))

#     return rmse


# # =========================
# # 主实验
# # =========================
# if __name__ == "__main__":

#     MODELS = ["LSTM", "GRU", "LSTM-SNP", "CBLSTM-SNP"]

#     results = []

#     os.makedirs("results", exist_ok=True)

#     for dataset in DATASETS:
#         print(f"\n===== DATASET: {dataset} =====")

#         row = [dataset]

#         for model_name in MODELS:
#             rmse = run_once(dataset, model_name)

#             print(f"{model_name}: RMSE = {rmse:.4f}")
#             row.append(rmse)

#         results.append(row)

#     # =========================
#     # 保存 CSV
#     # =========================
#     df = pd.DataFrame(results,
#                       columns=["Dataset", "LSTM", "GRU", "LSTM-SNP", "CBLSTM-SNP"])

#     save_path = "./results/rmse_results.csv"
#     df.to_csv(save_path, index=False)

#     print("\nSaved:", save_path)