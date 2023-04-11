import tensorflow as tf
import tensorflow_datasets as tfds

# shuffle_files = True: The MNIST data is only stored in a
# single file, but for larger datasets with multiple files on disk, it's good practice to shuffle them when training.

# as_supervised = True: Returns a tuple(img, label) instead of a dictionary {'image': img, 'label': label}.

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
