"""Official CIFAR-10 loading and split-specific preprocessing."""

import hashlib
import pickle
import tarfile

import numpy as np
import tensorflow as tf

ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
SPLIT_SEED = 2026
BATCH = 128


def load(archive, test=False):
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("CIFAR-10 archive checksum mismatch")
    names = ["test_batch"] if test else [f"data_batch_{i}" for i in range(1, 6)]
    images, labels = [], []
    with tarfile.open(archive, "r:gz") as bundle:
        for name in names:
            batch = pickle.load(bundle.extractfile(f"cifar-10-batches-py/{name}"),
                                encoding="bytes")
            images.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            labels.extend(batch[b"labels"])
    return np.concatenate(images), np.asarray(labels, dtype=np.int32)


def split(labels):
    rng = np.random.default_rng(SPLIT_SEED)
    by_class = [rng.permutation(np.flatnonzero(labels == label)) for label in range(10)]
    train = np.sort(np.concatenate([indices[500:] for indices in by_class]))
    validation = np.sort(np.concatenate([indices[:500] for indices in by_class]))
    return train, validation


def normalize_stats(images):
    flat = images.reshape(-1, 3)
    return [(flat.mean(axis=0) / 255).tolist(), (flat.std(axis=0) / 255).tolist()]


def batches(images, labels, stats, config=None, seed=0, epoch=0):
    with tf.device("/CPU:0"):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if config is not None:
        dataset = dataset.shuffle(len(labels), seed=seed * 1000 + epoch,
                                  reshuffle_each_iteration=False)
    mean, std = [tf.constant(values, dtype=tf.float32) for values in stats]
    axis = tf.range(32)

    def preprocess(index, batch):
        pixels, targets = batch
        key = tf.stack([tf.cast(seed * 1000 + epoch, index.dtype), index])
        keys = tf.random.experimental.stateless_split(key, 3)
        count = tf.shape(pixels)[0]
        if config is not None and config["augment"]:
            offsets = tf.random.stateless_uniform((count, 2), keys[0], 0, 9, tf.int32)
            flip = tf.random.stateless_uniform((count, 1, 1, 1), keys[1]) < 0.5
            pixels = tf.pad(pixels, [[0, 0], [4, 4], [4, 4], [0, 0]])
            pixels = tf.gather(pixels, offsets[:, :1] + axis, batch_dims=1, axis=1)
            pixels = tf.gather(pixels, offsets[:, 1:] + axis, batch_dims=1, axis=2)
            pixels = tf.where(flip, tf.reverse(pixels, [2]), pixels)
        pixels = (tf.cast(pixels, tf.float32) / 255 - mean) / std
        if config is not None and config["cutout"]:
            center = tf.random.stateless_uniform((count, 2), keys[2], 0, 32, tf.int32)
            rows = axis - center[:, :1]
            columns = axis - center[:, 1:]
            keep = (rows[:, :, None] < -8) | (rows[:, :, None] >= 8)
            keep = keep | (columns[:, None, :] < -8) | (columns[:, None, :] >= 8)
            pixels = tf.where(keep[..., None], pixels, 0.0)
        return pixels, targets

    return (dataset.batch(BATCH).enumerate()
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))
