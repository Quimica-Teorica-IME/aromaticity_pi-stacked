"""
Random-forest analysis of Q2 descriptor basis-set dependence.

Run from the directory containing norm.csv.  The script is
deliberately deterministic and writes numerical results and figures to
ML_Q2_analysis/.
"""

from pathlib import Path
import json
import joblib
import os
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
N_REPEATS = 20
N_SPLITS = 5
N_TREES = 500
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ML_Q2_analysis"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
MODELS = OUT / "models"
for folder in (RESULTS, FIGURES, MODELS):
	folder.mkdir(parents=True, exist_ok=True)

ID_COLUMNS = ["System", "Ring", "Basis Set"]
DESCRIPTORS = [
	"|Q2|_(ring atoms)", "Q2_(zz-ring atoms)", "|Q2|(0)", "Q2(0)zz",
	"|Q2|(1)", "Q2(1)zz", "|Q2|(-1)", "Q2(-1)zz",
]
ABS_DESCRIPTORS = [DESCRIPTORS[i] for i in (0, 2, 4, 6)]
DATASETS = {"norm": ROOT / "norm.csv", "non-norm": ROOT / "non-norm.csv"}
LABELS = {
	"|Q2|_(ring atoms)": r"$|Q_2|_{\mathrm{ring\ atoms}}$",
	"|Q2|(0)": r"$|Q_2|(0)$", "|Q2|(1)": r"$|Q_2|(1)$", "|Q2|(-1)": r"$|Q_2|(-1)$",
}


def make_preprocessor(features):
	return ColumnTransformer(
		[("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features)],
		remainder="drop",
	)


def make_regressor(features):
	return Pipeline([("encode", make_preprocessor(features)), ("model", RandomForestRegressor(
		n_estimators=N_TREES, min_samples_leaf=2, max_features=0.8,
		random_state=RANDOM_STATE, n_jobs=-1))])


def make_classifier():
	return Pipeline([("encode", make_preprocessor(["descriptor"])), ("model", RandomForestClassifier(
		n_estimators=N_TREES, min_samples_leaf=2, max_features="sqrt",
		random_state=RANDOM_STATE, n_jobs=-1))])


def metrics(y_true, prediction):
	return {"R2": r2_score(y_true, prediction),
			"RMSE": mean_squared_error(y_true, prediction) ** 0.5,
			"MAE": mean_absolute_error(y_true, prediction)}


def ordinary_splits(n):
	for repeat in range(N_REPEATS):
		splitter = KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
		yield repeat, list(splitter.split(np.arange(n)))


def grouped_splits(groups):
	groups = np.asarray(groups)
	unique = np.unique(groups)
	for repeat in range(N_REPEATS):
		shuffled = np.random.default_rng(RANDOM_STATE + repeat).permutation(unique)
		fold_ids = np.arange(len(shuffled)) % N_SPLITS
		fold_for_group = dict(zip(shuffled, fold_ids))
		test_fold = np.array([fold_for_group[group] for group in groups])
		yield repeat, [(np.flatnonzero(test_fold != fold), np.flatnonzero(test_fold == fold))
						for fold in range(N_SPLITS)]


def cross_validate_regression(data, target, features, split_kind):
	groups = data["Group"].to_numpy()
	splitter = ordinary_splits(len(data)) if split_kind == "ordinary" else grouped_splits(groups)
	rows = []
	for repeat, folds in splitter:
		for fold, (train, test) in enumerate(folds):
			model = make_regressor(features)
			model.fit(data.iloc[train][features], data.iloc[train][target])
			row = metrics(data.iloc[test][target], model.predict(data.iloc[test][features]))
			rows.append({"repeat": repeat + 1, "fold": fold + 1, "validation": split_kind,
						 "target": target, "predictors": "+".join(features), **row})
	return pd.DataFrame(rows)


def save_plot(fig, stem):
	fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
	fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
	plt.close(fig)


def safe_target(target):
	return {"|Q2|_(ring atoms)": "Q2_ring_atoms", "Q2_(zz-ring atoms)": "Q2_zz_ring_atoms",
			"Q2_(zz,ring atoms)": "Q2_zz_ring_atoms",
			"|Q2|(0)": "Q2_0", "Q2(0)zz": "Q2_0_zz", "|Q2|(1)": "Q2_1",
			"Q2(1)zz": "Q2_1_zz", "|Q2|(-1)": "Q2_minus1", "Q2(-1)zz": "Q2_minus1_zz"}[target]


def shap_analysis(data, dataset, target):
	features = ID_COLUMNS
	model = make_regressor(features)
	model.fit(data[features], data[target])
	transformed = model.named_steps["encode"].transform(data[features])
	names = list(model.named_steps["encode"].get_feature_names_out(features))
	explainer = shap.TreeExplainer(model.named_steps["model"])
	values = explainer.shap_values(transformed)
	values = values[0] if isinstance(values, list) else values
	grouped = {feature: [i for i, name in enumerate(names) if name.startswith(f"categorical__{feature}_")]
			   for feature in features}
	rows = []
	for feature, indices in grouped.items():
		# Sum absolute contributions across that variable's levels per observation,
		# then average observations; this preserves the original variable identity.
		importance = np.abs(values[:, indices]).sum(axis=1).mean()
		rows.append({"dataset": dataset, "target": target, "variable": feature,
					 "mean_abs_SHAP": importance})
	safe = safe_target(target)
	label = LABELS.get(target, target)
	fig = plt.figure(figsize=(7, 5))
	shap.summary_plot(values, transformed, feature_names=names, show=False, plot_size=None)
	plt.title(f"{dataset}: {label}")
	save_plot(fig, f"SHAP_summary_{dataset}_{safe}")
	grouped_df = pd.DataFrame(rows).sort_values("mean_abs_SHAP")
	joblib.dump(model, MODELS / f"RF_{dataset}_{safe}.joblib")
	fig, ax = plt.subplots(figsize=(7, 4.5))
	ax.barh(grouped_df["variable"], grouped_df["mean_abs_SHAP"], color=["#2a9d8f", "#e9c46a", "#e76f51"])
	ax.set_xlabel("Mean absolute SHAP value")
	ax.set_ylabel("Original predictor")
	ax.set_title(f"Grouped SHAP importance: {dataset}, {label}")
	save_plot(fig, f"SHAP_grouped_{dataset}_{safe}")
	return pd.DataFrame(rows)


def grouped_permutation(data, dataset, target):
	rows = []
	features = ID_COLUMNS
	for repeat, folds in grouped_splits(data["Group"]):
		for fold, (train, test) in enumerate(folds):
			model = make_regressor(features)
			model.fit(data.iloc[train][features], data.iloc[train][target])
			baseline = mean_squared_error(data.iloc[test][target], model.predict(data.iloc[test][features])) ** 0.5
			test_data = data.iloc[test][features].copy()
			rng = np.random.default_rng(RANDOM_STATE + repeat * 100 + fold)
			for variable in features:
				permuted = test_data.copy()
				permuted[variable] = rng.permutation(permuted[variable].to_numpy())
				score = mean_squared_error(data.iloc[test][target], model.predict(permuted)) ** 0.5
				rows.append({"dataset": dataset, "target": target, "variable": variable,
							 "repeat": repeat + 1, "fold": fold + 1,
							 "importance_RMSE_increase": score - baseline})
	result = pd.DataFrame(rows)
	summary = result.groupby(["dataset", "target", "variable"], as_index=False)["importance_RMSE_increase"].agg(
		permutation_mean="mean", permutation_std="std")
	summary["ranking"] = summary.groupby(["dataset", "target"])["permutation_mean"].rank(ascending=False, method="min").astype(int)
	plot_data = summary.sort_values("permutation_mean")
	fig, ax = plt.subplots(figsize=(7, 4.5))
	ax.barh(plot_data["variable"], plot_data["permutation_mean"], xerr=plot_data["permutation_std"], color="#457b9d", capsize=3)
	ax.set_xlabel("RMSE increase after joint permutation")
	ax.set_ylabel("Original predictor")
	ax.set_title(f"Grouped permutation importance: {dataset}, {LABELS.get(target, target)}")
	save_plot(fig, f"permutation_importance_{dataset}_{safe_target(target)}")
	return summary


def correlations(data, dataset, target):
	pivot = data.pivot_table(index="Group", columns="Basis Set", values=target, aggfunc="mean")
	rows = []
	for first in pivot.columns:
		for second in pivot.columns:
			paired = pivot[[first, second]].dropna()
			if first == second:
				pearson = spearman = 1.0
			else:
				pearson = paired[first].corr(paired[second], method="pearson")
				spearman = paired[first].corr(paired[second], method="spearman")
			rows.append({"dataset": dataset, "target": target, "basis_set_1": first,
						 "basis_set_2": second, "Pearson_r": pearson, "Spearman_rho": spearman,
						 "R2": pearson ** 2})
	result = pd.DataFrame(rows)
	for measure in ("Pearson_r", "Spearman_rho", "R2"):
		matrix = result.pivot(index="basis_set_1", columns="basis_set_2", values=measure)
		fig, ax = plt.subplots(figsize=(6, 5))
		sns.heatmap(matrix, annot=True, fmt=".2f", vmin=-1 if measure != "R2" else 0, vmax=1,
					cmap="vlag", square=True, ax=ax)
		ax.set_title(f"{dataset}: {LABELS.get(target, target)} ({measure})")
		save_plot(fig, f"correlation_{measure}_{dataset}_{safe_target(target)}")
	return result


def variance_analysis(data, dataset, target):
	pivot = data.pivot_table(index="Group", columns="Basis Set", values=target, aggfunc="mean")
	within = pivot.var(axis=1, ddof=1).mean()
	total = data[target].var(ddof=1)
	rows = []
	for group, values in pivot.iterrows():
		clean = values.dropna()
		rows.append({"dataset": dataset, "target": target, "metric": "mean_absolute_difference",
					 "value": np.abs(clean.to_numpy()[:, None] - clean.to_numpy()[None, :]).mean() / 2})
		rows.append({"dataset": dataset, "target": target, "metric": "coefficient_of_variation_across_basis_sets",
					 "value": clean.std(ddof=1) / abs(clean.mean()) if clean.mean() else np.nan})
		rows.append({"dataset": dataset, "target": target, "metric": "maximum_range_across_basis_sets",
					 "value": clean.max() - clean.min()})
	rows += [{"dataset": dataset, "target": target, "metric": "mean_within_group_basis_set_variance", "value": within},
			 {"dataset": dataset, "target": target, "metric": "total_variance", "value": total}]
	try:
		fitted = smf.mixedlm(f"Q2_value ~ C(Q2_basis)", data.rename(columns={target: "Q2_value", "Basis Set": "Q2_basis"}), groups=data["Group"]).fit(reml=True, disp=False)
		rows.append({"dataset": dataset, "target": target, "metric": "mixed_model_basis_set_fixed_effect_variance",
					 "value": float(np.var(fitted.fittedvalues))})
	except Exception:
		rows.append({"dataset": dataset, "target": target, "metric": "mixed_model_basis_set_fixed_effect_variance", "value": np.nan})
	return pd.DataFrame(rows)


def classification(data, dataset, target):
	rows = []
	clean = data[["Group", "Basis Set", target]].rename(columns={target: "descriptor"}).dropna()
	groups = clean["Group"].to_numpy()
	for repeat, folds in grouped_splits(groups):
		for fold, (train, test) in enumerate(folds):
			model = make_classifier()
			model.fit(clean.iloc[train][["descriptor"]], clean.iloc[train]["Basis Set"])
			prediction = model.predict(clean.iloc[test][["descriptor"]])
			rows.append({"dataset": dataset, "descriptor": target, "repeat": repeat + 1,
						 "fold": fold + 1, "accuracy": accuracy_score(clean.iloc[test]["Basis Set"], prediction)})
	result = pd.DataFrame(rows)
	summary = result.groupby(["dataset", "descriptor"], as_index=False)["accuracy"].agg(
		accuracy_mean="mean", accuracy_std="std")
	summary["n_basis_sets"] = clean["Basis Set"].nunique()
	summary["random_baseline"] = 1 / summary["n_basis_sets"]
	return summary


def main():
	all_cv, all_shap, all_perm, all_ablation = [], [], [], []
	all_classification, all_correlations, all_variance = [], [], []
	final_rows = []
	for dataset, path in DATASETS.items():
		data = pd.read_csv(path)
		missing = set(ID_COLUMNS + DESCRIPTORS) - set(data.columns)
		if missing:
			raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
		data["Group"] = data["System"].astype(str).str.strip() + "_" + data["Ring"].astype(str).str.strip()
		data[DESCRIPTORS] = data[DESCRIPTORS].apply(pd.to_numeric, errors="raise")
		for target in DESCRIPTORS:
			ordinary = cross_validate_regression(data, target, ID_COLUMNS, "ordinary")
			grouped = cross_validate_regression(data, target, ID_COLUMNS, "grouped")
			cv = pd.concat([ordinary, grouped], ignore_index=True)
			all_cv.append(cv.assign(dataset=dataset))
			shap_df = shap_analysis(data, dataset, target)
			all_shap.append(shap_df)
			perm_df = grouped_permutation(data, dataset, target)
			all_perm.append(perm_df)
			all_correlations.append(correlations(data, dataset, target))
			all_variance.append(variance_analysis(data, dataset, target))
			all_classification.append(classification(data, dataset, target))
			model_a = grouped[grouped["predictors"] == "+".join(ID_COLUMNS)]
			model_b = cross_validate_regression(data, target, ["System", "Ring"], "grouped")
			all_ablation.append(pd.DataFrame({"dataset": dataset, "target": target,
				"metric": ["R2", "RMSE", "MAE"],
				"model_A_mean": [model_a[x].mean() for x in ("R2", "RMSE", "MAE")],
				"model_B_mean": [model_b[x].mean() for x in ("R2", "RMSE", "MAE")]}))
			ablation_plot = all_ablation[-1]
			fig, ax = plt.subplots(figsize=(7, 4.5))
			x = np.arange(len(ablation_plot))
			width = 0.36
			ax.bar(x - width / 2, ablation_plot["model_A_mean"], width, label="System + Ring + Basis Set", color="#e76f51")
			ax.bar(x + width / 2, ablation_plot["model_B_mean"], width, label="System + Ring", color="#264653")
			ax.set_xticks(x, ablation_plot["metric"])
			ax.set_ylabel("Grouped-CV mean")
			ax.set_title(f"Ablation: {dataset}, {LABELS.get(target, target)}")
			ax.legend(frameon=False, fontsize=8)
			save_plot(fig, f"ablation_{dataset}_{safe_target(target)}")
			grouped_mean = model_a[["R2", "RMSE", "MAE"]].mean()
			ordinary_mean = ordinary[["R2", "RMSE", "MAE"]].mean()
			shap_values = shap_df.set_index("variable")["mean_abs_SHAP"]
			perm_values = perm_df.set_index("variable")["permutation_mean"]
			delta = all_ablation[-1].set_index("metric")
			final_rows.append({"Dataset": dataset, "Descriptor": target,
				"CV_R2": ordinary_mean["R2"], "CV_RMSE": ordinary_mean["RMSE"], "CV_MAE": ordinary_mean["MAE"],
				"Grouped_CV_R2": grouped_mean["R2"], "Grouped_CV_RMSE": grouped_mean["RMSE"], "Grouped_CV_MAE": grouped_mean["MAE"],
				"SHAP_System": shap_values["System"], "SHAP_Ring": shap_values["Ring"], "SHAP_BasisSet": shap_values["Basis Set"],
				"Permutation_System": perm_values["System"], "Permutation_Ring": perm_values["Ring"], "Permutation_BasisSet": perm_values["Basis Set"],
				"Delta_R2_BasisSet": delta.loc["R2", "model_A_mean"] - delta.loc["R2", "model_B_mean"]})
		abs_shap = pd.concat(all_shap, ignore_index=True)
		abs_shap = abs_shap[(abs_shap.dataset == dataset) & abs_shap.target.isin(ABS_DESCRIPTORS)]
		fig, ax = plt.subplots(figsize=(8, 5))
		for variable, part in abs_shap.groupby("variable"):
			values = part.set_index("target").reindex(ABS_DESCRIPTORS)["mean_abs_SHAP"]
			ax.plot([LABELS[x] for x in ABS_DESCRIPTORS], values, marker="o", label=variable)
		ax.set_ylabel("Mean absolute SHAP value")
		ax.set_title(f"Grouped basis-set importance comparison: {dataset}")
		ax.legend(frameon=False)
		save_plot(fig, f"SHAP_comparison_{dataset}")
	cv_result = pd.concat(all_cv, ignore_index=True)
	cv_result.to_csv(RESULTS / "CV_results.csv", index=False)
	group_columns = ["dataset", "target", "predictors", "validation"]
	cv_summary = cv_result.groupby(group_columns, as_index=False).agg(
		R2_mean=("R2", "mean"), R2_std=("R2", "std"),
		RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
		MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"))
	cv_summary.to_csv(RESULTS / "CV_summary.csv", index=False)
	pd.concat(all_shap, ignore_index=True).to_csv(RESULTS / "SHAP_importance.csv", index=False)
	pd.concat(all_perm, ignore_index=True).to_csv(RESULTS / "permutation_importance.csv", index=False)
	ablation_result = pd.concat(all_ablation, ignore_index=True)
	ablation_result["Delta_R2"] = np.where(ablation_result["metric"] == "R2", ablation_result["model_A_mean"] - ablation_result["model_B_mean"], np.nan)
	ablation_result["RMSE_improvement"] = np.where(ablation_result["metric"] == "RMSE", ablation_result["model_B_mean"] - ablation_result["model_A_mean"], np.nan)
	ablation_result.to_csv(RESULTS / "ablation_results.csv", index=False)
	pd.concat(all_classification, ignore_index=True).to_csv(RESULTS / "basis_set_classification.csv", index=False)
	pd.concat(all_correlations, ignore_index=True).to_csv(RESULTS / "correlations.csv", index=False)
	pd.concat(all_variance, ignore_index=True).to_csv(RESULTS / "variance_analysis.csv", index=False)
	final = pd.DataFrame(final_rows)
	order = {name: i for i, name in enumerate(DESCRIPTORS)}
	final["_order"] = final["Descriptor"].map(order)
	final = final.sort_values(["_order", "Dataset"]).drop(columns="_order")
	final.to_csv(RESULTS / "final_summary.csv", index=False)
	with open(RESULTS / "interpretation_report.txt", "w", encoding="utf-8") as report:
		report.write(make_report(final))
	with open(MODELS / "analysis_metadata.json", "w", encoding="utf-8") as metadata:
		json.dump({"random_state": RANDOM_STATE, "n_repeats": N_REPEATS, "n_splits": N_SPLITS,
				   "n_estimators": N_TREES, "primary_validation": "grouped System_Ring CV"}, metadata, indent=2)
	print(f"Completed. Results and figures were written to: {OUT}")


def make_report(final):
	lines = ["Q2 basis-set ML analysis interpretation", "", "Grouped repeated CV is the primary validation (20 repeats x 5 folds).", "Ordinary repeated CV is a comparator and can be optimistic because structures recur across folds.", ""]
	for dataset in DATASETS:
		part = final[final["Dataset"] == dataset].copy()
		part["_descriptor_key"] = part["Descriptor"].astype(str).str.strip().str.replace(",", "-", regex=False)
		lines.append(f"Dataset: {dataset}")
		for target in ABS_DESCRIPTORS:
			key = str(target).strip().replace(",", "-")
			matches = part[part["_descriptor_key"] == key]
			if matches.empty:
				available = ", ".join(part["Descriptor"].astype(str))
				raise ValueError(f"Report descriptor {target!r} was not found for {dataset}. Available: {available}")
			row = matches.iloc[0]
			lines.append(f"{target}: grouped R2={row.Grouped_CV_R2:.4f}; grouped Delta R2 from Basis Set={row.Delta_R2_BasisSet:.4f}; grouped SHAP Basis Set={row.SHAP_BasisSet:.4g}.")
		lines.append("")
	lines += ["Interpretation rules:", "1. Directly demonstrated: numerical grouped-CV metrics, ablation Delta R2/RMSE, grouped SHAP, and grouped permutation results.", "2. Consistent with statistical analysis: convergence of these independent indicators with correlation and variance results.", "3. Exploratory: reverse basis-set classification; it is not causal evidence.", "SHAP and permutation importance quantify predictive reliance, not proof of basis-set independence or physical causality."]
	return "\n".join(lines) + "\n"


if __name__ == "__main__":
	main()
