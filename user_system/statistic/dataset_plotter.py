import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Callable

def has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def savefig(path: Path):
    ensure_dir(path)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def get_pair_value(df, feature, a, b, col):
    row = df[(df["feature"] == feature) & (
        ((df["split_a"] == a) & (df["split_b"] == b)) |
        ((df["split_a"] == b) & (df["split_b"] == a))
    )]
    if not row.empty:
        return float(row.iloc[0][col])
    return np.nan

class PlotContext:
    def __init__(self, class_stats: pd.DataFrame, split_drift: pd.DataFrame, outdir: Path):
        class_stats = class_stats.copy()
        class_stats["class_name"] = class_stats["class_name"].astype(str)
        class_stats["split"] = class_stats["split"].astype(str)

        self.class_stats = class_stats
        self.split_drift = split_drift.copy()
        self.outdir = Path(outdir)

        self.df_all = self.class_stats[self.class_stats["split"] == "ALL"].copy()
        self.df_train = self.class_stats[self.class_stats["split"] == "train"].copy()
        self.df_val = self.class_stats[self.class_stats["split"] == "val"].copy()
        self.df_test = self.class_stats[self.class_stats["split"] == "test"].copy()


# Strategy Base Class
class PlotStrategy(ABC):
    @abstractmethod
    def render(self, ctx: PlotContext) -> List[Path]:
        """Return list of saved figure paths."""
        raise NotImplementedError

# Strategy Implementation: 1) Distribution
class DistributionPlots(PlotStrategy):
    def render(self, ctx: PlotContext) -> List[Path]:
        saved = []
        base = ctx.outdir / "01_distribution"

        # 1.1 Class instances bar
        if has_cols(ctx.df_all, ["class_name", "num_instances"]):
            bar_df = (ctx.df_all.groupby("class_name", as_index=False)["num_instances"]
                      .sum().sort_values("num_instances", ascending=False))
            plt.figure(figsize=(10, 6))
            plt.bar(bar_df["class_name"], bar_df["num_instances"])
            plt.title("Instances per Class (ALL)")
            plt.xlabel("Class"); plt.ylabel("Num Instances")
            plt.xticks(rotation=45, ha="right")
            p = base / "class_instances_bar.png"
            savefig(p); saved.append(p)

        # 1.2 Radar: images% vs instances% (Top-N)
        if has_cols(ctx.df_all, ["class_name", "images_pct", "instances_pct", "num_instances"]):
            top_n = 8
            tmp = (ctx.df_all.sort_values("num_instances", ascending=False)
                   .head(top_n)[["class_name", "images_pct", "instances_pct"]])
            labels = tmp["class_name"].tolist()
            theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            theta += theta[:1]
            imgs = tmp["images_pct"].tolist(); imgs += imgs[:1]
            inst = tmp["instances_pct"].tolist(); inst += inst[:1]
            plt.figure(figsize=(7, 7))
            ax = plt.subplot(111, polar=True)
            ax.plot(theta, imgs, linewidth=2)
            ax.fill(theta, imgs, alpha=0.1)
            ax.plot(theta, inst, linewidth=2, linestyle="--")
            ax.set_xticks(theta[:-1]); ax.set_xticklabels(labels)
            ax.set_title("Radar: Images% vs Instances% (Top-8)")
            ax.set_rlabel_position(0)
            p = base / "class_images_vs_instances_radar.png"
            savefig(p); saved.append(p)

        # 1.3 Split-wise stacked bar
        cs = ctx.class_stats
        if has_cols(cs, ["class_name", "split", "num_instances"]):
            pivot = (cs[cs["split"].isin(["train","val","test"])]
                     .groupby(["class_name","split"], as_index=False)["num_instances"]
                     .sum().pivot(index="class_name", columns="split", values="num_instances").fillna(0))
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
            classes = pivot.index.tolist()
            ind = np.arange(len(classes))
            plt.figure(figsize=(11, 7))
            bottom = np.zeros(len(classes))
            for sp in ["train", "val", "test"]:
                if sp in pivot.columns:
                    vals = pivot[sp].values
                    plt.bar(ind, vals, bottom=bottom, label=sp)
                    bottom += vals
            plt.xticks(ind, classes, rotation=45, ha="right")
            plt.ylabel("Num Instances"); plt.title("Split-wise Stacked Bar")
            plt.legend()
            p = base / "split_stacked_bar.png"
            savefig(p); saved.append(p)

        return saved


# Strategy Implementation: 2) Resolution/Format
class ResolutionPlots(PlotStrategy):
    def render(self, ctx: PlotContext) -> List[Path]:
        saved = []
        base = ctx.outdir / "02_resolution"

        # 2.1 Megapixels by split (boxplot)
        cs = ctx.class_stats
        if has_cols(cs, ["split", "med_megapixels"]):
            data = [ctx.df_train["med_megapixels"].dropna(),
                    ctx.df_val["med_megapixels"].dropna(),
                    ctx.df_test["med_megapixels"].dropna()]
            labels = ["train","val","test"]
            plt.figure(figsize=(8, 6))
            plt.boxplot([d.values for d in data], tick_labels=labels, showfliers=False)
            plt.title("Megapixels by Split (per-class medians)")
            plt.ylabel("Median Megapixels")
            p = base / "megapixels_box_by_split.png"
            savefig(p); saved.append(p)

        # 2.2 Width vs Height scatter (ALL)
        if has_cols(ctx.df_all, ["class_name","med_width","med_height","num_images"]):
            res_df = (ctx.df_all.groupby("class_name", as_index=False)
                      .agg({"med_width":"median","med_height":"median","num_images":"sum"}))
            plt.figure(figsize=(8, 6))
            sizes = 50 * (res_df["num_images"] / max(1, res_df["num_images"].max())).clip(lower=0.2)
            plt.scatter(res_df["med_width"], res_df["med_height"], s=sizes)
            for _, row in res_df.iterrows():
                plt.text(row["med_width"], row["med_height"], row["class_name"], fontsize=8)
            plt.title("Resolution Scatter (Median Width vs Height, ALL)")
            plt.xlabel("Median Width (px)"); plt.ylabel("Median Height (px)")
            p = base / "resolution_scatter.png"
            savefig(p); saved.append(p)

        # 2.3 Aspect ratio histogram (overlay by split)
        if has_cols(cs, ["split","med_aspect_ratio"]):
            plt.figure(figsize=(9, 6))
            for sp, df in [("train", ctx.df_train), ("val", ctx.df_val), ("test", ctx.df_test)]:
                vals = df["med_aspect_ratio"].dropna().values
                if len(vals) > 0:
                    plt.hist(vals, bins=30, alpha=0.5, density=True, label=sp)
            plt.title("Aspect Ratio Distribution by Split")
            plt.xlabel("Median Aspect Ratio"); plt.ylabel("Density"); plt.legend()
            p = base / "aspect_ratio_hist_by_split.png"
            savefig(p); saved.append(p)

        return saved


# Strategy Implementation: 3) Quality/Information
class QualityPlots(PlotStrategy):
    def render(self, ctx: PlotContext) -> List[Path]:
        saved = []
        base = ctx.outdir / "03_quality"
        cs = ctx.class_stats

        # 3.1 Entropy histogram overlay
        if has_cols(cs, ["split","med_entropy"]):
            plt.figure(figsize=(9, 6))
            for sp, df in [("train", ctx.df_train), ("val", ctx.df_val), ("test", ctx.df_test)]:
                vals = df["med_entropy"].dropna().values
                if len(vals) > 0:
                    plt.hist(vals, bins=30, alpha=0.5, density=True, label=sp)
            plt.title("Entropy Distribution by Split")
            plt.xlabel("Median Entropy"); plt.ylabel("Density"); plt.legend()
            p = base / "entropy_hist_by_split.png"
            savefig(p); saved.append(p)

        # 3.2 Blur variance boxplot
        if has_cols(cs, ["split","med_blur_var"]):
            data = [ctx.df_train["med_blur_var"].dropna().values,
                    ctx.df_val["med_blur_var"].dropna().values,
                    ctx.df_test["med_blur_var"].dropna().values]
            labels = ["train","val","test"]
            plt.figure(figsize=(8, 6))
            plt.boxplot(data, tick_labels=labels, showfliers=False)
            plt.title("Blur Variance by Split (per-class medians)")
            plt.ylabel("Median Blur Variance")
            p = base / "blur_var_box_by_split.png"
            savefig(p); saved.append(p)

        # 3.3 SNR histogram overlay
        if has_cols(cs, ["split","snr_db"]):
            plt.figure(figsize=(9, 6))
            for sp, df in [("train", ctx.df_train), ("val", ctx.df_val), ("test", ctx.df_test)]:
                vals = df["snr_db"].dropna().values
                if len(vals) > 0:
                    plt.hist(vals, bins=30, alpha=0.5, density=True, label=sp)
            plt.title("SNR (dB) Distribution by Split")
            plt.xlabel("SNR (dB)"); plt.ylabel("Density"); plt.legend()
            p = base / "snr_hist_by_split.png"
            savefig(p); saved.append(p)

        return saved


# Strategy Implementation: 4) Color/Lighting
class ColorPlots(PlotStrategy):
    def render(self, ctx: PlotContext) -> List[Path]:
        saved = []
        base = ctx.outdir / "04_color"
        cs = ctx.class_stats

        # 4.1 RGB mean scatter (R vs G)
        if has_cols(ctx.df_all, ["class_name","med_r_mean","med_g_mean"]):
            rgb_df = ctx.df_all[["class_name","med_r_mean","med_g_mean"]].dropna()
            plt.figure(figsize=(8, 6))
            plt.scatter(rgb_df["med_r_mean"], rgb_df["med_g_mean"])
            for _, row in rgb_df.iterrows():
                plt.text(row["med_r_mean"], row["med_g_mean"], row["class_name"], fontsize=8)
            plt.title("RGB Mean Scatter (R vs G, ALL medians per class)")
            plt.xlabel("Median R mean"); plt.ylabel("Median G mean")
            p = base / "rgb_mean_scatter_R_vs_G.png"
            savefig(p); saved.append(p)

        # 4.2 R/G/B mean box by split
        if has_cols(cs, ["split","med_r_mean","med_g_mean","med_b_mean"]):
            for ch, col in [("R","med_r_mean"), ("G","med_g_mean"), ("B","med_b_mean")]:
                data = [ctx.df_train[col].dropna().values,
                        ctx.df_val[col].dropna().values,
                        ctx.df_test[col].dropna().values]
                labels = ["train","val","test"]
                plt.figure(figsize=(8, 6))
                plt.boxplot(data, tick_labels=labels, showfliers=False)
                plt.title(f"{ch} Channel Mean by Split (per-class medians)")
                plt.ylabel(f"Median {ch} mean")
                p = base / f"{ch.lower()}_mean_box_by_split.png"
                savefig(p); saved.append(p)

        # 4.3 Gray mean histogram overlay
        if has_cols(cs, ["split","med_gray_mean"]):
            plt.figure(figsize=(9, 6))
            for sp, df in [("train", ctx.df_train), ("val", ctx.df_val), ("test", ctx.df_test)]:
                vals = df["med_gray_mean"].dropna().values
                if len(vals) > 0:
                    plt.hist(vals, bins=30, alpha=0.5, density=True, label=sp)
            plt.title("Gray Mean Distribution by Split")
            plt.xlabel("Median Gray Mean"); plt.ylabel("Density"); plt.legend()
            p = base / "gray_mean_hist_by_split.png"
            savefig(p); saved.append(p)

        return saved


# Strategy Implementation: 5) Geometry/Coverage
class GeometryPlots(PlotStrategy):
    def render(self, ctx: PlotContext) -> List[Path]:
        saved = []
        base = ctx.outdir / "05_geometry"

        # 5.1 Relative box area hist (ALL, per-class median)
        if has_cols(ctx.df_all, ["med_box_area_rel"]):
            vals = ctx.df_all["med_box_area_rel"].dropna().values
            if len(vals) > 0:
                plt.figure(figsize=(9, 6))
                plt.hist(vals, bins=30, alpha=0.8)
                plt.title("Relative Box Area (per-class median, ALL)")
                plt.xlabel("Median Box Area (relative)"); plt.ylabel("Count of Classes")
                p = base / "box_area_rel_hist_all.png"
                savefig(p); saved.append(p)

        # 5.2 Box aspect ratio boxplot
        cs = ctx.class_stats
        if has_cols(cs, ["split","med_box_aratio"]):
            data = [ctx.df_train["med_box_aratio"].dropna().values,
                    ctx.df_val["med_box_aratio"].dropna().values,
                    ctx.df_test["med_box_aratio"].dropna().values]
            labels = ["train","val","test"]
            plt.figure(figsize=(8, 6))
            plt.boxplot(data, tick_labels=labels, showfliers=False)
            plt.title("Box Aspect Ratio by Split (per-class medians)")
            plt.ylabel("Median Box Aspect Ratio")
            p = base / "box_aratio_box_by_split.png"
            savefig(p); saved.append(p)

        # 5.3 Density vs Coverage scatter (ALL)
        if has_cols(ctx.df_all, ["med_box_density_per_mp","med_fg_coverage_pct","class_name"]):
            geo_df = ctx.df_all[["class_name","med_box_density_per_mp","med_fg_coverage_pct"]].dropna()
            if len(geo_df) > 0:
                plt.figure(figsize=(8, 6))
                plt.scatter(geo_df["med_box_density_per_mp"], geo_df["med_fg_coverage_pct"])
                for _, row in geo_df.iterrows():
                    plt.text(row["med_box_density_per_mp"], row["med_fg_coverage_pct"],
                             row["class_name"], fontsize=8)
                plt.title("Density vs Coverage (per-class medians, ALL)")
                plt.xlabel("Median Box Density per MP"); plt.ylabel("Median Foreground Coverage (%)")
                p = base / "density_vs_coverage_scatter.png"
                savefig(p); saved.append(p)

        return saved


# Strategy Implementation: 6) Drift/Generalization
class DriftPlots(PlotStrategy):
    def render(self, ctx: PlotContext) -> List[Path]:
        saved = []
        base = ctx.outdir / "06_drift"
        sd = ctx.split_drift

        # 6.1 JS heatmap
        if has_cols(sd, ["feature","split_a","split_b","js_divergence"]):
            pairs = [("train","val"), ("train","test"), ("val","test")]
            pair_labels = [f"{a}-{b}" for a,b in pairs]
            features = sorted(sd["feature"].unique())
            heat_js = np.full((len(features), len(pairs)), np.nan)
            for i, feat in enumerate(features):
                for j, (a,b) in enumerate(pairs):
                    heat_js[i, j] = get_pair_value(sd, feat, a, b, "js_divergence")
            plt.figure(figsize=(9, max(4, len(features)*0.35)))
            im = plt.imshow(heat_js, aspect="auto")
            plt.title("Split Divergence (JS) by Feature")
            plt.xlabel("Split Pair"); plt.ylabel("Feature")
            plt.xticks(range(len(pair_labels)), pair_labels)
            plt.yticks(range(len(features)), features)
            plt.colorbar(im, label="JS Divergence")
            p = base / "split_drift_js_heatmap.png"
            savefig(p); saved.append(p)

        # 6.2 Wasserstein-1 heatmap
        if has_cols(sd, ["feature","split_a","split_b","wasserstein1"]):
            pairs = [("train","val"), ("train","test"), ("val","test")]
            pair_labels = [f"{a}-{b}" for a,b in pairs]
            features = sorted(sd["feature"].unique())
            heat_w = np.full((len(features), len(pairs)), np.nan)
            for i, feat in enumerate(features):
                for j, (a,b) in enumerate(pairs):
                    heat_w[i, j] = get_pair_value(sd, feat, a, b, "wasserstein1")
            plt.figure(figsize=(9, max(4, len(features)*0.35)))
            im = plt.imshow(heat_w, aspect="auto")
            plt.title("Split Divergence (Wasserstein-1) by Feature")
            plt.xlabel("Split Pair"); plt.ylabel("Feature")
            plt.xticks(range(len(pair_labels)), pair_labels)
            plt.yticks(range(len(features)), features)
            plt.colorbar(im, label="Wasserstein-1")
            p = base / "split_drift_wasserstein_heatmap.png"
            savefig(p); saved.append(p)

        # 6.3 Overlay hist (selected features)
        cs = ctx.class_stats
        features_for_overlay = [
            ("med_megapixels","Megapixels"),
            ("med_entropy","Entropy"),
            ("snr_db","SNR (dB)"),
        ]
        for col, label in features_for_overlay:
            if col in cs.columns:
                plt.figure(figsize=(9, 6))
                for sp, df in [("train", ctx.df_train), ("val", ctx.df_val), ("test", ctx.df_test)]:
                    vals = df[col].dropna().values
                    if len(vals) > 0:
                        plt.hist(vals, bins=30, alpha=0.5, density=True, label=sp)
                plt.title(f"{label} Distribution by Split (Overlay)")
                plt.xlabel(label); plt.ylabel("Density"); plt.legend()
                p = base / f"{col}_hist_overlay_by_split.png"
                savefig(p); saved.append(p)

        return saved


# Registry
class PlotRegistry:
    def __init__(self):
        self._strategies: Dict[str, PlotStrategy] = {}

    def register(self, name: str, strategy: PlotStrategy):
        self._strategies[name] = strategy

    def enable_all(self) -> List[str]:
        return list(self._strategies.keys())

    def get(self, name: str) -> PlotStrategy:
        return self._strategies[name]

    def names(self) -> List[str]:
        return list(self._strategies.keys())


# Facade
def plot_all(class_stats_csv: str, split_drift_csv: str, outdir: str,
             enabled: List[str] = None) -> List[Path]:
    class_stats = pd.read_csv(class_stats_csv)
    split_drift = pd.read_csv(split_drift_csv)
    ctx = PlotContext(class_stats, split_drift, Path(outdir))

    registry = PlotRegistry()
    registry.register("distribution", DistributionPlots())
    registry.register("resolution",   ResolutionPlots())
    registry.register("quality",      QualityPlots())
    registry.register("color",        ColorPlots())
    registry.register("geometry",     GeometryPlots())
    registry.register("drift",        DriftPlots())

    to_run = enabled if enabled else registry.enable_all()

    saved: List[Path] = []
    for name in to_run:
        strat = registry.get(name)
        saved.extend(strat.render(ctx))
    return saved


if __name__ == "__main__":
    CLASS_STATS_CSV = "user_system/statistic/unified_class_stats.csv"
    SPLIT_DRIFT_CSV = "user_system/statistic/unified_class_stats.csv"
    OUTDIR = "user_system/statistic/OUTPUT"


    paths = plot_all(CLASS_STATS_CSV, SPLIT_DRIFT_CSV, OUTDIR)
    print(f"Saved {len(paths)} figures.")
    for p in paths:
        print(p)
