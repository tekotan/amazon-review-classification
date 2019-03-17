import my_model
import argparse
import model_params


def train_model(model, restore_ckpt=False):
    params = model_params.params
    if model == "RNN":
        network = my_model.LSTMModel(params)

        model_op = my_model.RunModel(network, params)
        model_op.train(restore_ckpt)
    elif model == "CNN":
        network = my_model.ConvNetModel(params)

        model_op = my_model.RunModel(network, params)
        model_op.train()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("model")
    # args = parser.parse_args()
    train_model("RNN")
    # train_model("CNN")
