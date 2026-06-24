from __future__ import annotations
import math
import os
import pickle
import time
import random
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from pathlib import Path
from typing import Any
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier



def onehot(labels):
    """
    Convert integer class labels into one-hot vectors.
    """
    classes = sorted(set(labels))
    onehot_mapping = {
        class_id: np.identity(len(classes))[index, :]
        for index, class_id in enumerate(classes)
    }

    return np.array(
        list(map(onehot_mapping.get, labels)),
        dtype=np.int32,
    )


def load_data(d):
    """
    Load the hypergraph, features, and labels.

    Expected files:
        hypergraph.pickle
        feature_simplet.pickle
        labels.pickle
    """
    with open(os.path.join(d, "hypergraph.pickle"), "rb") as handle:
        hypergraph = pickle.load(handle)
        print("number of hyperedges is", len(hypergraph))

    with open(
        os.path.join(d, "feature_simplet.pickle"), "rb"
    ) as handle:
        features = pickle.load(handle).todense()

    with open(os.path.join(d, "labels.pickle"), "rb") as handle:
        labels = onehot(pickle.load(handle))

    return {
        "hypergraph": hypergraph,
        "features": features,
        "labels": labels,
        "n": features.shape[0],
    }


def extract_frequency(pattern, line_number=None):
    """
    Extract the frequency at the beginning of a pattern.

    Example:
        "4.0-[(0,0), (1,0)]-(1)" -> 4.0
    """
    frequency_text, separator, _ = pattern.partition("-")

    if not separator:
        location = (
            f" at mining-file line {line_number}"
            if line_number is not None
            else ""
        )
        raise ValueError(
            f"Cannot extract frequency from pattern{location}: {pattern!r}"
        )

    try:
        frequency = float(frequency_text)
    except ValueError as exception:
        location = (
            f" at mining-file line {line_number}"
            if line_number is not None
            else ""
        )
        raise ValueError(
            f"Invalid pattern frequency{location}: {frequency_text!r}"
        ) from exception

    if not math.isfinite(frequency) or frequency < 0:
        raise ValueError(
            f"Pattern frequency must be finite and non-negative, "
            f"got {frequency}"
        )

    return frequency


def read_pattern_records(pattern_file):
    """
    Read all patterns and their frequencies.

    Metadata before the separator line is skipped. Results are sorted
    from highest frequency to lowest frequency.

    Returns:
        List of (pattern_string, frequency) tuples.
    """
    records = []
    seen = set()
    reading_patterns = False

    with open(pattern_file, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            if line in seen:
                raise ValueError(
                    f"Duplicate pattern at mining-file line "
                    f"{line_number}: {line}"
                )

            frequency = extract_frequency(line, line_number)

            seen.add(line)
            records.append((line, frequency))

    # Python sorting is stable, so equal-frequency patterns retain
    # their original file order.
    records.sort(key=lambda record: record[1], reverse=True)

    return records


def read_patterns(pattern_file, min_frequency=0.0):
    """
    Read, sort, and filter patterns by frequency.

    Args:
        pattern_file: FreSCo mining-result file.
        min_frequency: Minimum frequency required for a pattern.

    Returns:
        patterns: surviving patterns sorted by decreasing frequency.
        frequencies: corresponding frequency list.
    """
    if min_frequency < 0:
        raise ValueError(
            f"min_frequency must be non-negative, got {min_frequency}"
        )

    records = read_pattern_records(pattern_file)

    # Create the sorted lists before filtering.
    sorted_patterns = [pattern for pattern, _ in records]
    sorted_frequencies = [frequency for _, frequency in records]

    surviving_patterns = []
    surviving_frequencies = []

    for pattern, frequency in zip(
        sorted_patterns,
        sorted_frequencies,
    ):
        if frequency >= min_frequency:
            surviving_patterns.append(pattern)
            surviving_frequencies.append(frequency)

    return surviving_patterns, surviving_frequencies


def read_vertex_images(image_file):
    """
    Read the vertex-image occurrence map.

    Expected format:
        pattern_string<TAB>vertex_id
    """
    pairs = []

    with open(image_file, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            try:
                pattern, vertex_text = line.rsplit("\t", maxsplit=1)
            except ValueError as exception:
                raise ValueError(
                    f"Invalid occMap line {line_number}: expected "
                    "'<pattern>\\t<vertex id>'"
                ) from exception

            pattern = pattern.strip()

            try:
                vertex = int(vertex_text.strip())
            except ValueError as exception:
                raise ValueError(
                    f"Invalid vertex ID at occMap line {line_number}: "
                    f"{vertex_text!r}"
                ) from exception

            pairs.append((pattern, vertex))

    return pairs


def build_sparse_matrix(patterns, vertex_pairs, num_vertices):
    """
    Construct sparse matrix A where:

        A[i, j] = 1

    iff vertex i appears as an image of pattern j.
    """
    if num_vertices < 0:
        raise ValueError(
            f"num_vertices must be non-negative, got {num_vertices}"
        )

    pattern_to_col = {
        pattern: column
        for column, pattern in enumerate(patterns)
    }

    rows = []
    cols = []

    for pattern, vertex in vertex_pairs:
        if pattern not in pattern_to_col:
            raise ValueError(
                "Occurrence map references an unexpected pattern: "
                f"{pattern}"
            )

        if not 0 <= vertex < num_vertices:
            raise IndexError(
                f"Vertex ID {vertex} is outside [0, {num_vertices})"
            )

        rows.append(vertex)
        cols.append(pattern_to_col[pattern])

    values = np.ones(len(rows), dtype=np.float32)

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(num_vertices, len(patterns)),
        dtype=np.float32,
    )

    # Duplicate coordinates are summed by CSR construction.
    # Convert the matrix back to binary values.
    matrix.sum_duplicates()
    matrix.data[:] = 1.0

    return matrix


def build_feature_matrix(
    data,
    pattern_file,
    image_file,
    min_frequency=0.0,
):
    """
    Build a feature matrix using patterns meeting min_frequency.

    Columns are sorted by decreasing pattern frequency.

    Returns:
        matrix:
            CSR matrix shaped
            (number_of_vertices, number_of_surviving_patterns).

        patterns:
            Surviving patterns in matrix-column order.

        frequencies:
            Frequency of each surviving pattern in column order.
    """
    all_records = read_pattern_records(pattern_file)

    all_patterns = {
        pattern
        for pattern, _ in all_records
    }

    patterns = [
        pattern
        for pattern, frequency in all_records
        if frequency >= min_frequency
    ]

    frequencies = [
        frequency
        for _, frequency in all_records
        if frequency >= min_frequency
    ]

    surviving_pattern_set = set(patterns)
    vertex_pairs = read_vertex_images(image_file)

    filtered_vertex_pairs = []

    for pattern, vertex in vertex_pairs:
        if pattern not in all_patterns:
            raise ValueError(
                "Occurrence map references a pattern that is not "
                f"present in the mining file: {pattern}"
            )

        # Ignore occurrences belonging to patterns removed by the
        # minimum-frequency threshold.
        if pattern in surviving_pattern_set:
            filtered_vertex_pairs.append((pattern, vertex))

    num_vertices = int(data["features"].shape[0])

    matrix = build_sparse_matrix(
        patterns,
        filtered_vertex_pairs,
        num_vertices,
    )

    return matrix, patterns, frequencies

def as_float(value):
    """Convert a Python number or one-element tensor to a Python float."""
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return float(value)

def scalar(value):
    """Convert a scalar tensor or numeric value to float."""
    if torch.is_tensor(value):
        return value.detach().cpu().item()

    return float(value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_matrix(threshold, pattern_file, image_file):
    """
    Load the feature matrix saved for one frequency threshold.
    """
    feature_matrix, _, _ = build_feature_matrix(
        data=dataset,
        pattern_file=pattern_file,
        image_file=image_file,
        min_frequency=threshold,
    )

    return feature_matrix.tocsr()

def train_for_threshold(
    threshold,
    Y,
    G,
    pattern_file,
    image_file,
    args,
):
    feature_matrix = load_feature_matrix(
        threshold,
        pattern_file,
        image_file,
    )

    number_of_patterns = feature_matrix.shape[1]

    X = torch.as_tensor(
        feature_matrix.toarray(),
        dtype=torch.float32,
        device=device,
    )

    final_train_accuracies = []
    mean_epoch_train_accuracies = []
    final_test_accuracies = []
    best_test_accuracies = []

    for run in range(1, args.n_runs + 1):
        set_seed(args.seed + run - 1)
        args.split = run

        _, train_idx, test_idx = data.load(args)

        train_idx = torch.as_tensor(
            train_idx,
            dtype=torch.long,
            device=device,
        )
        test_idx = torch.as_tensor(
            test_idx,
            dtype=torch.long,
            device=device,
        )

        model, optimizer = initialise(X, Y, G, args)

        epoch_train_accuracies = []
        epoch_test_accuracies = []

        start_time = time.time()

        for epoch in range(args.epochs):
            # Training
            model.train()
            optimizer.zero_grad()

            output = model(X)
            loss = F.nll_loss(
                output[train_idx],
                Y[train_idx],
            )

            loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()

            with torch.no_grad():
                output = model(X)

                train_accuracy = scalar(
                    accuracy(
                        output[train_idx],
                        Y[train_idx],
                    )
                )

                test_accuracy = scalar(
                    accuracy(
                        output[test_idx],
                        Y[test_idx],
                    )
                )

            epoch_train_accuracies.append(train_accuracy)
            epoch_test_accuracies.append(test_accuracy)

        final_train_accuracy = epoch_train_accuracies[-1]
        final_test_accuracy = epoch_test_accuracies[-1]
        best_test_accuracy = max(epoch_test_accuracies)

        final_train_accuracies.append(final_train_accuracy)
        mean_epoch_train_accuracies.append(
            np.mean(epoch_train_accuracies)
        )
        final_test_accuracies.append(final_test_accuracy)
        best_test_accuracies.append(best_test_accuracy)

        print(
            f"threshold={threshold}, "
            f"run={run}/{args.n_runs}, "
            f"patterns={number_of_patterns}, "
            f"train={final_train_accuracy:.4f}, "
            f"test={final_test_accuracy:.4f}, "
            f"best test={best_test_accuracy:.4f}, "
            f"time={time.time() - start_time:.2f}s"
        )

        del optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "threshold": int(threshold),
        "number_of_patterns": int(number_of_patterns),

        "mean_final_train_accuracy": float(
            np.mean(final_train_accuracies)
        ),
        "std_final_train_accuracy": float(
            np.std(final_train_accuracies)
        ),
        "mean_epoch_train_accuracy": float(
            np.mean(mean_epoch_train_accuracies)
        ),

        "mean_final_test_accuracy": float(
            np.mean(final_test_accuracies)
        ),
        "std_final_test_accuracy": float(
            np.std(final_test_accuracies)
        ),
        "mean_best_test_accuracy": float(
            np.mean(best_test_accuracies)
        ),
        "std_best_test_accuracy": float(
            np.std(best_test_accuracies)
        ),
    }, model

"""Visualize FreSCo vertex-pattern features with label-colored t-SNE."""


def labels_to_class_ids(labels: Any) -> np.ndarray:
    """
    Convert labels into one class ID per vertex.

    Accepted inputs:
    - Shape (num_vertices,): integer class IDs.
    - Shape (num_vertices, num_classes): one-hot rows.
    """
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels.astype(np.int64, copy=False)

    if labels.ndim != 2:
        raise ValueError(
            "labels must have shape (num_vertices,) or "
            "(num_vertices, num_classes)"
        )

    positive_counts = np.count_nonzero(labels, axis=1)
    invalid_rows = np.flatnonzero(positive_counts != 1)
    if invalid_rows.size:
        preview = invalid_rows[:10].tolist()
        raise ValueError(
            "Each vertex must have exactly one label. Invalid rows: "
            f"{preview}"
        )

    return np.argmax(labels, axis=1).astype(np.int64)


def compute_tsne(
    feature_matrix: Any,
    labels: Any,
    *,
    perplexity: float = 30.0,
    svd_components: int = 50,
    max_samples: int | None = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """
    Return (embedding, sampled_class_ids, vertex_indices, silhouette).

    TruncatedSVD first reduces a high-dimensional sparse feature matrix. The
    silhouette score is computed before t-SNE because t-SNE can exaggerate
    visual cluster separation.
    """
    class_ids = labels_to_class_ids(labels)
    num_vertices, num_features = feature_matrix.shape

    if class_ids.shape[0] != num_vertices:
        raise ValueError(
            f"Matrix has {num_vertices} vertices but labels has "
            f"{class_ids.shape[0]} rows"
        )
    if num_vertices < 3:
        raise ValueError("t-SNE requires at least three vertices")
    if num_features == 0:
        raise ValueError("The feature matrix has no pattern columns")

    vertex_indices = np.arange(num_vertices)
    if max_samples is not None and num_vertices > max_samples:
        if max_samples < 3:
            raise ValueError("max_samples must be at least 3")

        unique_labels, label_counts = np.unique(
            class_ids, return_counts=True
        )
        can_stratify = (
            unique_labels.size <= max_samples
            and unique_labels.size <= num_vertices - max_samples
            and np.all(label_counts >= 2)
        )
        if can_stratify:
            vertex_indices, _ = train_test_split(
                vertex_indices,
                train_size=max_samples,
                stratify=class_ids,
                random_state=random_state,
            )
        else:
            random = np.random.default_rng(random_state)
            vertex_indices = random.choice(
                vertex_indices,
                size=max_samples,
                replace=False,
            )

        vertex_indices = np.sort(vertex_indices)
        feature_matrix = feature_matrix[vertex_indices]
        class_ids = class_ids[vertex_indices]
        num_vertices = max_samples
        print(f"Using {num_vertices} sampled vertices for t-SNE")

    max_components = min(
        svd_components,
        num_vertices - 1,
        num_features - 1,
    )
    if max_components >= 2:
        reducer = TruncatedSVD(
            n_components=max_components,
            random_state=random_state,
        )
        reduced = reducer.fit_transform(feature_matrix)
        explained = reducer.explained_variance_ratio_.sum()
        print(
            f"SVD dimensions: {num_features} -> {max_components}; "
            f"explained variance: {explained:.3f}"
        )
    else:
        reduced = (
            feature_matrix.toarray()
            if sparse.issparse(feature_matrix)
            else np.asarray(feature_matrix)
        )

    unique_labels, label_counts = np.unique(class_ids, return_counts=True)
    can_score = (
        unique_labels.size > 1
        and unique_labels.size < num_vertices
        and np.all(label_counts >= 2)
    )
    score = (
        float(silhouette_score(reduced, class_ids))
        if can_score
        else None
    )

    effective_perplexity = min(
        float(perplexity),
        max(1.0, (num_vertices - 1) / 3.0),
    )
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca" if reduced.shape[1] >= 2 else "random",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(reduced)

    return embedding, class_ids, vertex_indices, score


def plot_tsne(
    feature_matrix: Any,
    labels: Any,
    *,
    output_file: str | Path | None = None,
    perplexity: float = 30.0,
    svd_components: int = 50,
    max_samples: int | None = 5000,
    random_state: int = 42,
    show: bool = True,
    mode: str = "labeled",
):
    plt.rcParams.update({
        "font.size": 20,          # default font size
        "axes.titlesize": 20,     # title
        "axes.labelsize": 20,     # axis labels
        # "xtick.labelsize": 14,    # x tick labels
        # "ytick.labelsize": 14,    # y tick labels
        # "legend.fontsize": 14,    # legend entries
        "legend.title_fontsize": 20,
    })
    """Compute and plot a label-colored t-SNE embedding."""
    embedding, class_ids, vertex_indices, score = compute_tsne(
        feature_matrix,
        labels,
        perplexity=perplexity,
        svd_components=svd_components,
        max_samples=max_samples,
        random_state=random_state,
    )

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.set_axis_off()
    for class_id in np.unique(class_ids):
        mask = class_ids == class_id
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=0.75,
            label=f"Label {class_id}",
        )

    # title = f"t-SNE of {mode}-based features"
    # if score is not None:
    #     title += f"\nPre-t-SNE silhouette by label: {score:.3f}"
    #     print(f"Pre-t-SNE silhouette score by label: {score:.3f}")

    # axis.set_title(title)
    # axis.set_xlabel("t-SNE 1")
    # axis.set_ylabel("t-SNE 2")
    # axis.legend(title="Vertex label", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=1200, bbox_inches="tight", format = "pdf")
        print(f"Saved plot to {output_file}")

    if show:
        plt.show()

    return embedding, vertex_indices, figure, axis

def get_split(Y, p=0.2):
    # Same validation/test split helper used in train_val.py.
    from random import sample, shuffle

    Y = Y.tolist()
    N, nclass = len(Y), len(set(Y))
    D = [[] for _ in range(nclass)]

    for i, y in enumerate(Y):
        D[y].append(i)

    k = int(N * p / nclass)
    val_idx = torch.cat(
        [torch.LongTensor(sample(idxs, k)) for idxs in D]
    ).tolist()
    test_idx = list(set(range(N)) - set(val_idx))

    return val_idx, test_idx


def train_and_extract_features(feature_matrix, Y, G, args):
    """
    Train UniGCNII exactly as train_val.py does, then return the
    final learned vertex feature matrix from every run.

    The features are the input to model.convs[-1], i.e. the tensor
    immediately before the final Linear classifier.
    """
    X = torch.FloatTensor(feature_matrix.toarray()).cuda()

    test_accs = []
    best_val_accs = []
    best_test_accs = []
    run_results = []
    running_times = []
    for run in range(1, args.n_runs + 1):
        run_dir = out_dir / f"{run}"
        run_dir.makedirs_p()

        # Same split handling as train_val.py.
        args.split = run
        _, train_idx, test_idx = data.load(args)
        val_idx, test_idx = get_split(Y[test_idx], 0.2)

        train_idx = torch.LongTensor(train_idx).cuda()
        val_idx = torch.LongTensor(val_idx).cuda()
        test_idx = torch.LongTensor(test_idx).cuda()

        model, optimizer = initialise(X, Y, G, args)

        baselogger.info(
            f"Run {run}/{args.n_runs}, Total Epochs: {args.epochs}"
        )
        baselogger.info(model)
        baselogger.info(
            f"total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        tic_run = time.time()
        best_val_acc, best_test_acc = 0, 0
        test_acc, Z, bad_counter = 0, None, 0
        start_time = time.perf_counter()
        for epoch in range(args.epochs):
            tic_epoch = time.time()
            model.train()

            optimizer.zero_grad()
            Z = model(X)
            loss = F.nll_loss(Z[train_idx], Y[train_idx])

            loss.backward()
            optimizer.step()

            train_time = time.time() - tic_epoch

            model.eval()
            Z = model(X)
            train_acc = accuracy(Z[train_idx], Y[train_idx])
            test_acc = accuracy(Z[test_idx], Y[test_idx])
            val_acc = accuracy(Z[val_idx], Y[val_idx])

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter >= args.patience:
                    break

            baselogger.info(
                f"epoch:{epoch} | loss:{loss:.4f} | "
                f"train acc:{train_acc:.2f} | val acc:{val_acc:.2f} | "
                f"best_test_acc: {best_test_acc:.2f} | "
                f"test acc:{test_acc:.2f} | "
                f"time:{train_time * 1000:.1f}ms"
            )

        running_times.append(time.perf_counter() - start_time)
        captured = {}

        def capture_classifier_input(module, inputs):
            captured["features"] = inputs[0].detach().cpu().clone()

        hook = model.convs[-1].register_forward_pre_hook(
            capture_classifier_input
        )

        model.eval()
        model(X)
        hook.remove()

        resultlogger.info(
            f"Run {run}/{args.n_runs}, best test accuracy: "
            f"{best_test_acc:.2f}, acc(last): {test_acc:.2f}, "
            f"total time: {time.time() - tic_run:.2f}s"
        )

        test_accs.append(test_acc)
        best_val_accs.append(best_val_acc)
        best_test_accs.append(best_test_acc)

        run_results.append({
            "run": run,
            "features": captured["features"],
            "last_test_accuracy": test_acc,
            "best_validation_accuracy": best_val_acc,
            "best_test_accuracy": best_test_acc,
            "epochs_trained": epoch + 1,
        })

    resultlogger.info(
        f"Average final test accuracy: {np.mean(test_accs)} "
        f"± {np.std(test_accs)}"
    )
    resultlogger.info(
        f"Average best test accuracy: {np.mean(best_test_accs)} "
        f"± {np.std(best_test_accs)}"
    )

    return run_results, model, running_times

def as_sklearn_feature_matrix(feature_matrix):
    """
    Convert a SciPy, NumPy, or PyTorch vertex-feature matrix into a
    scikit-learn-compatible two-dimensional matrix. CSR is retained
    for simplet features to avoid making their mostly-zero matrices dense.
    """
    if torch.is_tensor(feature_matrix):
        feature_matrix = feature_matrix.detach().cpu().numpy()

    if sparse.issparse(feature_matrix):
        return feature_matrix.tocsr()

    feature_matrix = np.asarray(feature_matrix)
    if feature_matrix.ndim != 2:
        raise ValueError(
            "feature_matrix must have shape (num_vertices, num_features)"
        )

    return feature_matrix


def features_for_run(feature_source, run):
    """
    Return the appropriate feature matrix for one run. Raw simplet
    matrices are shared across runs; UniGCNII embeddings are stored
    separately because each trained model produces different embeddings.
    """
    if isinstance(feature_source, dict):
        if run not in feature_source:
            raise KeyError(f"No UniGCNII features were supplied for run {run}")
        return feature_source[run]

    return feature_source


def make_classifiers(seed):
    """
    Create fresh estimators for one run. Scaling without centering keeps
    sparse simplet matrices sparse. Five neighbours balances local detail
    with some robustness to one unusual training vertex.
    """
    sparse_scaler = StandardScaler(with_mean=False)

    return {
        "Logistic regression": Pipeline([
            ("scale", sparse_scaler),
            ("model", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=seed,
            )),
        ]),
        "Linear SVM": Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("model", LinearSVC(
                C=1.0,
                max_iter=10000,
                random_state=seed,
            )),
        ]),
        "SVM": Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("model", SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                random_state=seed,
            )),
        ]),
        "KNN": Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("model", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1,
            )),
        ]),
        "Decision tree": DecisionTreeClassifier(
            random_state=seed,
        ),
    }


def evaluate_classical_models(feature_sets, Y, args):
    """
    Fit logistic regression, a linear SVM, KNN, and a decision tree for
    every feature set and every args.split run. The train/test indices are
    loaded exactly once per run and reused by all feature sets and models,
    making the comparisons directly comparable.

    Returns:
        results_df: one row per feature set, classifier, and run.
        predictions: test-set predictions keyed by
            (feature_set, classifier, run).
    """
    if torch.is_tensor(Y):
        y = Y.detach().cpu().numpy()
    else:
        y = np.asarray(Y)

    if y.ndim == 2:
        y = y.argmax(axis=1)
    if y.ndim != 1:
        raise ValueError("Y must contain one class label per vertex")

    records = []
    predictions = {}

    for run in range(1, args.n_runs + 1):
        args.split = run
        _, train_idx, test_idx = data.load(args)
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        for feature_name, feature_source in feature_sets.items():
            X = as_sklearn_feature_matrix(
                features_for_run(feature_source, run)
            )

            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"{feature_name} has {X.shape[0]} rows, but Y has "
                    f"{y.shape[0]} labels"
                )

            for classifier_name, classifier in make_classifiers(
                args.seed + run - 1
            ).items():
                start_time = time.time()
                classifier.fit(X[train_idx], y[train_idx])

                train_prediction = classifier.predict(X[train_idx])
                test_prediction = classifier.predict(X[test_idx])

                train_accuracy = accuracy_score(
                    y[train_idx],
                    train_prediction,
                )
                test_accuracy = accuracy_score(
                    y[test_idx],
                    test_prediction,
                )

                records.append({
                    "feature_set": feature_name,
                    "classifier": classifier_name,
                    "run": run,
                    "number_of_features": X.shape[1],
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "fit_and_predict_seconds": time.time() - start_time,
                })

                predictions[(feature_name, classifier_name, run)] = {
                    "vertex_indices": test_idx,
                    "y_true": y[test_idx],
                    "y_pred": test_prediction,
                }

                print(
                    f"{feature_name} | {classifier_name} | "
                    f"run {run}/{args.n_runs} | "
                    f"train={train_accuracy:.4f} | "
                    f"test={test_accuracy:.4f}"
                )

    return pd.DataFrame(records), predictions
