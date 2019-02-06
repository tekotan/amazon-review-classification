import my_model
import argparse
import ipdb

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

network = my_model.LSTMModel(params, predict=True)
train = my_model.RunModel(network, params, predict=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("msg")
    args = parser.parse_args()
    ipdb.set_trace()
    print(train.predict(args.msg))
