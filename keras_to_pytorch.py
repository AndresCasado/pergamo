import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import tqdm

from encoder.pose_encoder_10D_torch import PoseEncoder10D


def check_same_input_output(
        keras_layer,
        pytorch_module,
        *input_size,
) -> torch.Tensor:
    x_np = np.random.randn(*input_size).astype(np.float)
    x_tf = tf.constant(x_np, dtype=tf.float32)
    x_pt = torch.from_numpy(x_np).float()

    y_tf = keras_layer(x_tf)
    y_pt = pytorch_module(x_pt)

    diff = y_pt - y_tf.numpy()
    return diff.norm()


def read_keras_create_pytorch():
    pose_encoder_keras = tf.keras.models.load_model('encoder/pose_encoder_10D', compile=False)

    pose_encoder_layer = pose_encoder_keras.get_layer('pose_encoder')
    dense = pose_encoder_layer.get_layer('dense')
    dense_1 = pose_encoder_layer.get_layer('dense_1')

    dense_w, dense_b = dense.weights
    dense_w = dense_w.numpy().T
    dense_b = dense_b.numpy()

    dense_1_w, dense_1_b = dense_1.weights
    dense_1_w = dense_1_w.numpy().T
    dense_1_b = dense_1_b.numpy()

    mypose_encoder = PoseEncoder10D()
    mypose_encoder.dense.weight = nn.Parameter(torch.from_numpy(dense_w))
    mypose_encoder.dense.bias = nn.Parameter(torch.from_numpy(dense_b))

    mypose_encoder.dense_1.weight = nn.Parameter(torch.from_numpy(dense_1_w))
    mypose_encoder.dense_1.bias = nn.Parameter(torch.from_numpy(dense_1_b))

    model_path = 'pose_encoder_10D_converted.pth'

    torch.save(mypose_encoder.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    errors = []
    for i in tqdm.tqdm(range(1000)):
        with torch.no_grad():
            diff_norm = check_same_input_output(
                pose_encoder_keras,
                mypose_encoder,
                3, 69,
            )

        errors.append(diff_norm)
    errors = torch.stack(errors)
    print(f'Mean error: {errors.mean()}')


if __name__ == '__main__':
    read_keras_create_pytorch()
