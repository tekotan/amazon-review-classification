from my_model import Params

params = Params(
    batch_size=256,
    dropout_keep_prob=1,
    learning_rate=0.008,
    report_step=100,
    save_step=100,
    num_epochs=10,
    lstmUnits=[16],
    fc_layer_units=[6],
    output_classes=6,
)
