from SentimentAnalysis.my_model import Params

params = Params(
    batch_size=256,
    dropout_keep_prob=1,
    learning_rate=0.00007,
    report_step=10,
    save_step=100,
    num_epochs=1,
    lstmUnits=[256, 256, 128],
    fc_layer_units=[128, 64, 32],
    output_classes=1,
)
