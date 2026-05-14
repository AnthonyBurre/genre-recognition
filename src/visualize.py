"""Exploratory visualization of the feature set.

Where ``evaluate.py`` answers "how good is a trained model", this module
answers the question that comes *before* modeling: "does the feature set in
``features.py`` actually carry genre signal at all?" A 3D PCA scatter,
color-coded by genre, is the quickest read - tight per-genre clusters mean
the features separate genres; one uniform blob means they don't.

It's deliberately a peer of ``data.py`` / ``features.py`` / ``evaluate.py``
with its own ``python -m src.visualize`` CLI, rather than a flag on the
training driver - exploration happens off to the side of a run.

Adding another visualization is a few lines: write a ``plot_*`` function with
the ``(fm, out_dir, tag, genres) -> None`` signature and add it to
``PLOT_REGISTRY``. The CLI picks it up automatically.
"""
import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import data as data_mod
from . import features as features_mod
from .config import DATASETS, DEFAULT_DATASET, RANDOM_SEED, RESULTS_DIR, Dataset, get_dataset
from .features import FeatureMatrix

# Hand-picked accent colors for GTZAN's 10 genres - the first two match the
# accent colors used in ``evaluate.py``. Other datasets fall back to a
# qualitative colormap (see ``_genre_colors``).
_GTZAN_GENRE_COLORS: dict[str, str] = {
    "blues": "#2a9d8f",
    "classical": "#e76f51",
    "country": "#e9c46a",
    "disco": "#264653",
    "hiphop": "#f4a261",
    "jazz": "#8ab17d",
    "metal": "#9b5de5",
    "pop": "#f15bb5",
    "reggae": "#00bbf9",
    "rock": "#bc6c25",
}


def _genre_colors(genres: tuple[str, ...]) -> dict[str, str]:
    """A fixed genre -> color map for the given genre list.

    Uses the hand-picked GTZAN palette when every genre is covered by it;
    otherwise derives position-based colors from a qualitative colormap so
    any dataset's genre vocabulary renders consistently.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    if all(g in _GTZAN_GENRE_COLORS for g in genres):
        return {g: _GTZAN_GENRE_COLORS[g] for g in genres}
    cmap = plt.get_cmap("tab10" if len(genres) <= 10 else "tab20")
    return {g: to_hex(cmap(i % cmap.N)) for i, g in enumerate(genres)}


def _load_features(dataset: Dataset, split: str, fold: str) -> FeatureMatrix:
    """Featurize the requested data slice.

    ``fold="all"`` featurizes the full dataset index; otherwise the named
    fold of the requested split variant. Extraction is joblib-cached, so the
    first call is slow and every call after is fast.
    """
    if fold == "all":
        manifest = data_mod.build_index(dataset)
    else:
        split_obj = data_mod.get_or_build_split(dataset, variant=split)
        manifest = getattr(split_obj, fold)
    return features_mod.extract(manifest, sr=dataset.sample_rate, desc=f"features:{split}:{fold}")


def plot_pca3d(
    fm: FeatureMatrix, out_dir: Path, tag: str, genres: tuple[str, ...],
) -> None:
    """3D PCA scatter, one point per track, colored by genre.

    Features are z-scored before PCA - the set mixes wildly different scales
    (tempo ~120, MFCC moments near zero), so raw PCA would just recover the
    tempo axis. Same rationale as ``data.py``'s z-scored cosine distances.

    Writes both a static PNG (fixed viewing angle, for run artifacts) and an
    interactive HTML (rotatable, for actually reading cluster separation).
    """
    Xz = StandardScaler().fit_transform(fm.X)
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    coords = pca.fit_transform(Xz)
    evr = pca.explained_variance_ratio_
    genre_labels = np.array([genres[int(i)] for i in fm.y])
    genre_colors = _genre_colors(genres)

    axis_labels = [f"PC{i + 1} ({evr[i] * 100:.1f}%)" for i in range(3)]
    title = (
        f"PCA of feature set - {tag}  "
        f"(total variance explained: {evr.sum() * 100:.1f}%)"
    )

    # Static PNG - one scatter call per genre keeps the legend clean.
    import matplotlib.pyplot as plt  # noqa: F401  (registers 3D projection)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    for genre in genres:
        mask = genre_labels == genre
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            s=22, alpha=0.75, color=genre_colors[genre], label=genre,
        )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    png_path = out_dir / f"pca3d_{tag}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Interactive HTML - rotatable, which is what makes 3D actually readable.
    import plotly.express as px

    html_path = out_dir / f"pca3d_{tag}.html"
    plot_df = {
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "PC3": coords[:, 2],
        "genre": genre_labels,
        "track_id": fm.track_ids,
    }
    plot_fig = px.scatter_3d(
        plot_df, x="PC1", y="PC2", z="PC3",
        color="genre", color_discrete_map=genre_colors,
        category_orders={"genre": list(genres)},
        hover_data=["track_id"], title=title,
    )
    plot_fig.update_traces(marker=dict(size=4, opacity=0.8))
    plot_fig.update_layout(scene=dict(
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        zaxis_title=axis_labels[2],
    ))
    plot_fig.write_html(html_path)

    print(f"[visualize] wrote {png_path}")
    print(f"[visualize] wrote {html_path}")


# name -> plot function. Adding a visualization = add a function above and an
# entry here; the CLI's --plot choices derive from this dict.
PLOT_REGISTRY: dict[str, Callable[[FeatureMatrix, Path, str, tuple[str, ...]], None]] = {
    "pca3d": plot_pca3d,
}


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exploratory visualization of the audio feature set."
    )
    parser.add_argument("--plot", default="pca3d", choices=sorted(PLOT_REGISTRY),
                        help="Which visualization to render.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(DATASETS),
                        help="Which dataset to visualize.")
    parser.add_argument("--split", default="naive", choices=["naive", "filtered"],
                        help="Split variant to draw the fold from. Ignored when "
                             "--fold all.")
    parser.add_argument("--fold", default="train",
                        choices=["train", "val", "test", "all"],
                        help="Which data slice to visualize. 'all' uses the full "
                             "dataset index regardless of --split.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. Defaults to results/visualizations/.")
    args = parser.parse_args(argv)

    dataset = get_dataset(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{dataset.name}_{args.split}_{args.fold}" if args.fold != "all" else f"{dataset.name}_all"
    fm = _load_features(dataset, args.split, args.fold)
    print(f"[visualize] {args.plot} on {len(fm.y)} tracks ({tag})")
    PLOT_REGISTRY[args.plot](fm, out_dir, tag, dataset.genres)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
