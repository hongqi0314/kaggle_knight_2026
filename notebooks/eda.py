"""EDA notebook for Kaggle Knight 2026 - Insurance Premium Prediction."""

import polars as pl

DATA_DIR = "data/raw"

# ── Load data ────────────────────────────────────────────────────────────────
train = pl.read_csv(f"{DATA_DIR}/train.csv")
test = pl.read_csv(f"{DATA_DIR}/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Target column: Premium Amount")

# ── Schema & dtypes ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COLUMN TYPES")
print("=" * 60)
for name, dtype in zip(train.columns, train.dtypes):
    print(f"  {name:<25} {str(dtype)}")

# ── Numeric summary ─────────────────────────────────────────────────────────
numeric_cols = [
    c for c, d in zip(train.columns, train.dtypes) if d in (pl.Float64, pl.Int64)
]
print("\n" + "=" * 60)
print("NUMERIC SUMMARY (train)")
print("=" * 60)
print(train.select(numeric_cols).describe())

# ── Target distribution ─────────────────────────────────────────────────────
target = train["Premium Amount"]
print("\n" + "=" * 60)
print("TARGET: Premium Amount")
print("=" * 60)
print(f"  Mean:   {target.mean():.2f}")
print(f"  Median: {target.median():.2f}")
print(f"  Std:    {target.std():.2f}")
print(f"  Min:    {target.min()}")
print(f"  Max:    {target.max()}")
print(f"  Skew:   {target.skew():.4f}")

percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
for p in percentiles:
    val = target.quantile(p)
    print(f"  P{int(p*100):02d}:    {val:.2f}")

# ── Missing values ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
for dataset_name, df in [("Train", train), ("Test", test)]:
    null_counts = df.null_count()
    print(f"\n  {dataset_name}:")
    for col in df.columns:
        nc = null_counts[col][0]
        if nc > 0:
            pct = nc / len(df) * 100
            print(f"    {col:<25} {nc:>8} ({pct:.2f}%)")

# ── Categorical columns ────────────────────────────────────────────────────
cat_cols = [
    c
    for c, d in zip(train.columns, train.dtypes)
    if d == pl.String and c not in ("id", "Policy Start Date")
]
print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS - VALUE COUNTS")
print("=" * 60)
for col in cat_cols:
    vc = train[col].value_counts().sort("count", descending=True)
    print(f"\n  {col} (n_unique={train[col].n_unique()}):")
    for row in vc.iter_rows():
        val, cnt = row
        pct = cnt / len(train) * 100
        print(f"    {str(val):<25} {cnt:>8} ({pct:.1f}%)")

# ── Target by categorical ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREMIUM AMOUNT BY CATEGORICAL FEATURES")
print("=" * 60)
for col in cat_cols:
    agg = (
        train.group_by(col)
        .agg(
            pl.col("Premium Amount").mean().alias("mean"),
            pl.col("Premium Amount").median().alias("median"),
            pl.col("Premium Amount").std().alias("std"),
            pl.col("Premium Amount").count().alias("count"),
        )
        .sort("mean", descending=True)
    )
    print(f"\n  {col}:")
    for row in agg.iter_rows(named=True):
        print(
            f"    {str(row[col]):<25} mean={row['mean']:>10.2f}  "
            f"median={row['median']:>10.2f}  std={row['std']:>10.2f}  "
            f"n={row['count']}"
        )

# ── Correlations with target ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("CORRELATIONS WITH Premium Amount")
print("=" * 60)
num_cols_no_id = [c for c in numeric_cols if c not in ("id", "Premium Amount")]
corrs = []
for col in num_cols_no_id:
    r = train.select(pl.corr("Premium Amount", col)).item()
    corrs.append((col, r))
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for col, r in corrs:
    print(f"  {col:<25} {r:>8.4f}")

# ── Numeric feature distributions ──────────────────────────────────────────
print("\n" + "=" * 60)
print("NUMERIC FEATURE DISTRIBUTIONS")
print("=" * 60)
for col in num_cols_no_id:
    s = train[col].drop_nulls()
    print(
        f"  {col:<25} mean={s.mean():>10.2f}  "
        f"std={s.std():>10.2f}  "
        f"min={s.min():>10}  max={s.max():>10}  "
        f"nulls={train[col].null_count()}"
    )

# ── Policy Start Date analysis ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("POLICY START DATE")
print("=" * 60)
dates = train.with_columns(pl.col("Policy Start Date").str.to_datetime())
print(f"  Min date: {dates['Policy Start Date'].min()}")
print(f"  Max date: {dates['Policy Start Date'].max()}")

# Extract date features and check correlation with target
dates_with_feats = dates.with_columns(
    pl.col("Policy Start Date").dt.year().alias("year"),
    pl.col("Policy Start Date").dt.month().alias("month"),
    pl.col("Policy Start Date").dt.weekday().alias("dow"),
)
for feat in ["year", "month", "dow"]:
    r = dates_with_feats.select(pl.corr("Premium Amount", feat)).item()
    print(f"  Corr with {feat}: {r:.4f}")

# Monthly average premium
monthly = (
    dates_with_feats.group_by("year", "month")
    .agg(pl.col("Premium Amount").mean().alias("mean_premium"))
    .sort("year", "month")
)
print("\n  Monthly avg premium:")
for row in monthly.iter_rows(named=True):
    print(f"    {row['year']}-{row['month']:02d}: {row['mean_premium']:.2f}")

# ── Train vs Test distribution comparison ───────────────────────────────────
print("\n" + "=" * 60)
print("TRAIN vs TEST DISTRIBUTION COMPARISON")
print("=" * 60)
shared_numeric = [
    c for c in num_cols_no_id if c in test.columns and c != "Premium Amount"
]
for col in shared_numeric:
    tr = train[col].drop_nulls()
    te = test[col].drop_nulls()
    print(
        f"  {col:<25} "
        f"train_mean={tr.mean():>10.2f} test_mean={te.mean():>10.2f}  "
        f"train_std={tr.std():>10.2f} test_std={te.std():>10.2f}"
    )

# ── Duplicate check ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)
feature_cols = [c for c in train.columns if c not in ("id", "Premium Amount")]
n_dup = len(train) - train.select(feature_cols).n_unique()
print(f"  Duplicate feature rows in train: {n_dup}")

# ── Outlier detection (IQR) on target ────────────────────────────────────────
print("\n" + "=" * 60)
print("TARGET OUTLIERS (IQR method)")
print("=" * 60)
q1 = target.quantile(0.25)
q3 = target.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
n_outliers = train.filter(
    (pl.col("Premium Amount") < lower) | (pl.col("Premium Amount") > upper)
).shape[0]
print(f"  Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
print(f"  Bounds: [{lower:.2f}, {upper:.2f}]")
print(f"  Outliers: {n_outliers} ({n_outliers/len(train)*100:.2f}%)")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
