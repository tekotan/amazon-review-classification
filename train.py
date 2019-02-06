import my_model
import argparse


def train_model(model):
    params = my_model.Params(
        batch_size=128,
        dropout_keep_prob=1,
        learning_rate=0.01,
        report_step=5,
        save_step=100,
        num_epochs=2,
        lstmUnits=[1],
        fc_layer_units=[6],
        output_classes=6,
    )
    if model == "RNN":
        network = my_model.LSTMModel(params)

        model_op = my_model.RunModel(network, params)
        model_op.train()
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
