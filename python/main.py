from tensorflow.python.keras.layers.preprocessing.image_preprocessing import Rescaling
import image_functions
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def compute_threshold(true_labels, predicted):
    test_points = np.arange(0, 1, 0.01)
    auc = []
    for point in test_points:
        pred_labels = [1 if t >= point else 0 for t in predicted]
        auc.append(roc_auc_score(true_labels, pred_labels))
    return 0.01 * auc.index(max(auc))

seed = 123
train_ds, val_ds, test_ds = image_functions.create_dataset(seed)

_INPUT_DIMS = list(train_ds.as_numpy_iterator())[0][0][0].shape

# base_model = keras.applications.ResNet50(
#     include_top=False, weights="imagenet", input_shape=_INPUT_DIMS
# )
# base_model.trainable = False
# inputs = keras.Input(shape=_INPUT_DIMS)

# inference_mode = base_model(inputs, training=False)

# inference_mode = keras.layers.GlobalAveragePooling2D()(inference_mode)
# inference_mode = keras.layers.Dropout(0.2)(inference_mode)

# outputs = tf.keras.Sequential(
#     [
#         layers.Dense(64, activation="relu"),
#         layers.Dense(32, activation="relu"),
#         layers.Dense(16, activation="relu"),
#         layers.Dense(1, activation="sigmoid"),
#     ]
# )(inference_mode)
# model = keras.Model(inputs, outputs)

model = tf.keras.Sequential(
    [
        layers.InputLayer(input_shape=_INPUT_DIMS),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

metrics = {"AUC": [], "ACCURACY": [], "CONFUSION_MATRIX": []}

model.compile(
    optimizer=keras.optimizers.SGD(1e-4),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()],
)

first_history = model.fit(train_ds, epochs=10, validation_data=val_ds)

# base_model.trainable = True

# model.compile(
#     optimizer=keras.optimizers.SGD(1e-5),
#     loss=keras.losses.BinaryCrossentropy(),
#     metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()],
# )

# second_history = model.fit(train_ds, epochs=5, validation_data=val_ds)
# # t??h??n trainille j-statistiikka

train_predictions = model.predict(train_ds)
val_predictions = model.predict(val_ds)
predictions = np.concatenate([train_predictions, val_predictions])
train_labels = np.concatenate([y for x, y in train_ds], axis=0)
val_labels = np.concatenate([y for x, y in val_ds], axis=0)
labels = np.concatenate([train_labels, val_labels])
threshold = compute_threshold(labels, predictions)

labels = np.concatenate([y for x, y in test_ds], axis=0)

predictions = model.predict(test_ds)
pred_labels = [1 if t >= threshold else 0 for t in predictions]

metrics["AUC"].append(roc_auc_score(labels, predictions))
metrics["ACCURACY"].append(accuracy_score(labels, pred_labels))
metrics["CONFUSION_MATRIX"].append(confusion_matrix(labels, pred_labels))
print(f"AUC: {roc_auc_score(labels, predictions)}")
print(f"Binary accuracy: {accuracy_score(labels, pred_labels)}")
print(f"Confusion matrix:\n{confusion_matrix(labels, pred_labels)}")
print()
