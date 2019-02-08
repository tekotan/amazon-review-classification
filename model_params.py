from my_model import Params

params = Params(
    batch_size=32,
    dropout_keep_prob=0.8,
    learning_rate=0.005,
    report_step=5,
    save_step=10,
    num_epochs=2,
    lstmUnits=[1],
    fc_layer_units=[6],
    output_classes=6,
)
