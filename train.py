import my_model
conv_params = my_model.Params(learning_rate=0.01, filter_sizes=[
    2, 3, 4], dropout_keep_prob=1, report_step=10, total_iterations=100, batch_size=32, num_filters=1)
train_model = my_model.TrainModel
Model = my_model.ConvNetModel(conv_params)

train_model.train(Model, con_params)
