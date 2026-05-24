#!/usr/bin/env python3
import argparse
import time
import struct
import numpy as np
from sklearn.datasets import make_classification


def generate_dataset(n_samples, n_features, n_classes=2, random_seed=42):
    n_info = max(1, min(n_features - 1, n_features // 2))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_info,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_seed,
    )
    return X.astype(np.float32), y.astype(np.float32)


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


def save_csv(path, data, labels=None):
    if labels is not None:
        arr = np.column_stack([data, labels])
    else:
        arr = data
    np.savetxt(path, arr, delimiter=',', fmt='%.6f')


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
    total = args.n + q

    t0 = time.time()
    X, y = generate_dataset(total, args.d, args.classes, args.seed)
    gen_time = time.time() - t0

    X_train, y_train = X[:args.n], y[:args.n]
    X_query = X[args.n:]

    print(f"Generated: train={X_train.shape}, query={X_query.shape}, "
          f"classes={args.classes}, time={gen_time:.3f}s")

    train_path = f"{args.output}_train.npy"
    query_path = f"{args.output}_query.npy"
    save_binary(train_path, X_train, y_train)
    save_binary(query_path, X_query)
    print(f"Saved: {train_path}, {query_path}")

    save_csv(f"{args.output}_train.csv", X_train, y_train)
    save_csv(f"{args.output}_query.csv", X_query)
    print(f"Saved: {args.output}_train.csv, {args.output}_query.csv")


if __name__ == '__main__':
    main()
