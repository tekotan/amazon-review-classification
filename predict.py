import my_model
import argparse
import ipdb
import model_params

params = model_params.params
network = my_model.LSTMModel(params, predict=True)
train = my_model.RunModel(network, params, predict=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--review", type=str, required=True,)
    args = parser.parse_args()
    ipdb.set_trace()
    print(train.predict(args.review))
