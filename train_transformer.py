import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from helpers import *
from model import build_model, build_lstm_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

train_x = pd.read_csv(Path(processed_path, dataset_name))
train_x.dropna(inplace=True)
train_y = train_x[["price_change"]]
train_x = train_x[[x for x in train_x.columns if x not in exclude_for_x]]

train_x = np.expand_dims(train_x[list(train_x.columns)[:-1]].values, -2)
train_y = np.expand_dims(train_y.values, -1)
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

input_shape = train_x.shape[1:]
print(input_shape)

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="mse",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["mape", "mae"],
)
model.summary()

callbacks = [
    ReduceLROnPlateau(patience=30)
]

print("training...")

try:
    model.fit(
        train_x,
        train_y,
        validation_split=0.2,
        epochs=1000,
        batch_size=4,
        callbacks=callbacks,
    )
except KeyboardInterrupt:
    print("terminating training..")

model.save(Path(data_root, "models", "transformer"))

print("evaluating...")

model.evaluate(test_x, test_y, verbose=1)

