from my_model import Params

params = Params(
    batch_size=32,
    dropout_keep_prob=1,
    learning_rate=0.02,
    report_step=1,
    save_step=30,
    num_epochs=1000,
    lstmUnits=[2],
    fc_layer_units=[6],
    output_classes=6,
)
