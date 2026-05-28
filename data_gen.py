#!/usr/bin/env python3
import argparse
import time
import struct
import numpy as np


def generate_dataset(n_train, n_query, n_features, n_classes=2, random_seed=42):
    rng = np.random.default_rng(random_seed)
    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    y_train = rng.integers(0, n_classes, n_train).astype(np.float32)
    X_query = rng.standard_normal((n_query, n_features)).astype(np.float32)
    return X_train, y_train, X_query


def save_binary(path, data, labels=None):
    with open(path, 'wb') as f:
        if labels is not None:
            N, D = data.shape
            f.write(struct.pack('ii', N, D))
            f.write(data.tobytes())
            f.write(labels.tobytes())
        else:
            Q, D = data.shape
            f.write(struct.pack('ii', Q, D))
            f.write(data.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic datasets for KNN benchmarking'
    )
    parser.add_argument('--n', type=int, required=True,
                        help='Number of training samples')
    parser.add_argument('--d', type=int, required=True,
                        help='Number of features')
    parser.add_argument('--classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--q', type=int, default=0,
                        help='Number of query samples (default: n//5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path prefix')
    args = parser.parse_args()

    assert args.n >= 100, "n must be >= 100"
    assert args.d >= 2, "d must be >= 2"
    assert args.d < args.n, "d must be < n"

    q = args.q if args.q > 0 else max(args.n // 5, 100)

    t0 = time.time()
    X_train, y_train, X_query = generate_dataset(args.n, q, args.d, args.classes, args.seed)
    gen_time = time.time() - t0

    print(f"Generated: train={X_train.shape}, query={X_query.shape}, "
          f"classes={args.classes}, time={gen_time:.3f}s")

    train_path = f"{args.output}_train.npy"
    query_path = f"{args.output}_query.npy"
    save_binary(train_path, X_train, y_train)
    save_binary(query_path, X_query)
    print(f"Saved: {train_path}, {query_path}")


if __name__ == '__main__':
    main()
