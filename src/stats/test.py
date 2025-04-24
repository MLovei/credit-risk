from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from IPython.display import Markdown, display
from matplotlib.colors import ListedColormap
from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr, shapiro



CITRINE = "#e9cb0c"
NAPLES = "#ffd470"
CREAM = "#f3f6cb"
APPLE = "#9ea300"
MOSS = "#555610"
OLIVE = "#907E08"
HARVEST = "#E49F00"
PEAR = "#D1DC3A"
BACKGROUND_COLOR = "white"
ml_colors = [MOSS, APPLE, CREAM, NAPLES, CITRINE]
full_pallet = [APPLE, CITRINE, CREAM, HARVEST, OLIVE, NAPLES, MOSS, PEAR]
cmap = ListedColormap(ml_colors)


def shapiro_test_by_tag(
	df: pd.DataFrame,
	tag_col: str, value_col: str,
	display_output: bool = True
) -> Optional[Dict[str, Tuple[float, float]]]:
	"""Shapiro-Wilk tests on value_col grouped by tag_col,
	optionally display."""
	group_true = df.loc[df[tag_col] == True, value_col]
	group_false = df.loc[df[tag_col] == False, value_col]
	stat_true, p_value_true = shapiro(group_true)
	stat_false, p_value_false = shapiro(group_false)

	results = {
		"tag_true" : (stat_true, p_value_true),
		"tag_false": (stat_false, p_value_false),
	}

	if display_output:
		display(Markdown("### Shapiro-Wilk Test Results:"))
		for tag, (stat, p_value) in results.items():
			tag_value = "True" if tag == "tag_true" else "False"
			tag_label = f"{tag_col} = {tag_value}"
			display(Markdown(f"**For {tag_label}:**"))
			display(Markdown(f"* Test Statistic: {stat:.3f}"))
			display(Markdown(f"* P-value: {p_value:.3f}"))
		return None
	else:
		return results


def calculate_confidence_interval(data, confidence_level=0.95):
	"""Calc. CI for mean of data (Z-interval approx)."""
	mean = data.mean()
	sem = stats.sem(data)
	if pd.isna(sem) or sem == 0:
		return mean, mean, mean
	alpha = 1 - confidence_level
	z_critical = stats.norm.ppf(1 - alpha / 2)
	interval = sem * z_critical
	lower_ci = mean - interval
	upper_ci = mean + interval
	return mean, lower_ci, upper_ci


def non_parametric_test_by_tag(
	df: pd.DataFrame,
	tag_col: str,
	value_col: str,
	confidence_level: float = 0.95,
	display_output: bool = True,
) -> Optional[Dict[str, Dict[str, float]]]:
	"""MWU test & CI for value_col grouped by tag_col, optionally display."""
	group_true = df.loc[df[tag_col] == True, value_col]
	group_false = df.loc[df[tag_col] == False, value_col]
	statistic, p_value = stats.mannwhitneyu(group_true, group_false)

	mean_true, lower_ci_true, upper_ci_true = calculate_confidence_interval(
		group_true, confidence_level
	)
	mean_false, lower_ci_false, upper_ci_false = calculate_confidence_interval(
		group_false, confidence_level
	)

	results = {
		"tag_true"           : {
			"mean"    : mean_true,
			"lower_ci": lower_ci_true,
			"upper_ci": upper_ci_true,
		},
		"tag_false"          : {
			"mean"    : mean_false,
			"lower_ci": lower_ci_false,
			"upper_ci": upper_ci_false,
		},
		"non_parametric_test": {
			"statistic": statistic,
			"p_value"  : p_value,
		},
	}

	if display_output:
		display(
			Markdown(
				"### Non-parametric Test Results (Mann-Whitney U) "
				"and Confidence Intervals:"
			)
		)
		display(
			Markdown(f"**Confidence Level:** {confidence_level * 100:.0f}%")
		)
		for tag_key, ci_data in results.items():
			if tag_key in ["tag_true", "tag_false"]:
				tag_value = "True" if tag_key == "tag_true" else "False"
				tag_label = f"{tag_col} = {tag_value}"
				display(Markdown(f"**For {tag_label}:**"))
				display(Markdown(f"* Mean Value: {ci_data["mean"]:.2f}"))
				display(
					Markdown(
						f"* Confidence Interval: ({ci_data["lower_ci"]:.2f},"
						f" {ci_data["upper_ci"]:.2f})"
					)
				)
		np_results = results["non_parametric_test"]
		display(Markdown(f"**Mann-Whitney U Test:**"))
		display(Markdown(f"* p-value = {np_results["p_value"]:.3f}"))
		return None
	else:
		return results


def cramers_v(x, y):
	confusion_matrix = pd.crosstab(x, y)
	chi2 = chi2_contingency(confusion_matrix)[0]
	n = confusion_matrix.sum().sum()
	r, k = confusion_matrix.shape
	return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))


def categorical_feature_tests(
	df: pd.DataFrame,
	cat_col: str,
	target_col: str,
	significance_level: float = 0.05,
	display_output: bool = True,
) -> Optional[Dict[str, float]]:
	"""
	Performs Chi-Squared test and calculates Cramer's V for a categorical column
	against a binary target column. Optionally displays the results.

	Args:
		df (pd.DataFrame): Input DataFrame containing the data.
		cat_col (str): Name of the categorical column.
		target_col (str): Name of the binary target column.
		significance_level (float): Significance threshold for p-value.
		display_output (bool): Whether to display the results. Defaults to True.

	Returns:
		Optional[Dict[str, float]]: Dictionary of test results
		if display_output is False;
									otherwise None.
	"""
	contingency_table = pd.crosstab(df[cat_col], df[target_col])

	chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)

	results = {"chi2_statistic": chi2_statistic, "p_value": p_value}

	if p_value < significance_level:
		results["cramers_v"] = cramers_v(df[cat_col], df[target_col])

	if display_output:
		display(Markdown(f"**`{cat_col}` vs `{target_col}`**"))
		display(Markdown(f"* Chi-Squared Statistic:"
                         f" {results['chi2_statistic']:.3f}"))
		if p_value < significance_level:
			display(Markdown(f"* p-value: {results['p_value']:.3f} "
                             f"**(< {significance_level:.4f})**"))
		else:
			display(Markdown(f"* p-value: {results['p_value']:.3f} "
                             f"**(> {significance_level:.4f})**"))

		if p_value < significance_level:
			display(Markdown(
				f"*Significant association found* "
				f"**(Cramer's V: {results.get('cramers_v', 'N/A'):.3f})**"))
		else:
			display(Markdown(
				f"**No significant association found "
				f"(p >= {significance_level:.4f})**"))

		return None

	return results

def binary_heatmap(df: pd.DataFrame, target_col: str, cmap: str = cmap):
	"""
	Generates a heatmap of point biserial correlations between a
	target boolean column
	and all other numeric columns in a DataFrame,
	ranked by absolute correlation size.

	Args:
	  df: pandas DataFrame containing the data.
	  target_col: Name of the boolean target column in the DataFrame.
	  cmap: The colormap for the heatmap.

	Returns:
	  A matplotlib Axes object containing the heatmap.
	"""

	y = df[target_col]
	numeric_df = df.select_dtypes(include=["number"])

	if target_col in numeric_df:
		numeric_df = numeric_df.drop(target_col, axis=1)

	correlations = numeric_df.apply(lambda x: pointbiserialr(x, y)[0])
	correlations = correlations.abs().sort_values(ascending=False)

	plt.figure(figsize=(10, 8))
	sns.heatmap(
		correlations.to_frame(), annot=True, cmap=cmap, vmin=-1, vmax=1
	)
	plt.title(f"Point Biserial Correlation with {target_col} (Ranked)")
	plt.show()


def analyze_feature_correlations(
	df: pd.DataFrame,
	target: str = "TARGET",
	top_n: int = 10
) -> pd.DataFrame:
	"""
	Analyzes correlations between features and the target variable, combining
	both numerical and categorical correlations.

	Args:
		df (pd.DataFrame): Cleaned DataFrame.
		target (str): Target variable column name.
		top_n (int): Number of top positive and negative correlations to return.

	Returns:
		pd.DataFrame: DataFrame with top correlated features.
	"""
	df_copy = df.copy()

	numerical_cols = df_copy.select_dtypes(include=["float", "int"]).columns
	numerical_cols = [col for col in numerical_cols if col != target]

	corr_numerical = []
	for col in numerical_cols:
		if df_copy[col].nunique() > 1:
			corr, p_value = stats.pearsonr(
				df_copy[col].fillna(0), df_copy[target]
			)
			corr_numerical.append((col, corr, p_value))

	corr_numerical_df = pd.DataFrame(
		corr_numerical, columns=["Feature", "Correlation", "P_Value"]
	)

	categorical_cols = df_copy.select_dtypes(
		include=["category", "object"]
	).columns
	categorical_cols = [col for col in categorical_cols if col != target]

	corr_categorical = []
	for col in categorical_cols:
		if df_copy[col].nunique() > 1:
			cramers_val = cramers_v(df_copy[col], df_copy[target])
			contingency_table = pd.crosstab(df_copy[col], df_copy[target])
			chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
			corr_categorical.append((col, cramers_val, p_value))

	corr_categorical_df = pd.DataFrame(
		corr_categorical, columns=["Feature", "Correlation", "P_Value"]
	)

	corr_combined_df = pd.concat(
		[corr_numerical_df, corr_categorical_df], ignore_index=True
	)

	significant_features = corr_combined_df[
		corr_combined_df["P_Value"] < 0.05
		]

	top_positive = significant_features[
		significant_features["Correlation"] > 0
		].nlargest(top_n, "Correlation")
	top_negative = significant_features[
		significant_features["Correlation"] < 0
		].nsmallest(top_n, "Correlation")

	top_corr_df = pd.concat([top_positive, top_negative]).sort_values(
		"Correlation", ascending=False
	)

	top_corr_df["Importance"] = abs(top_corr_df["Correlation"]) / abs(
		top_corr_df["Correlation"].max()
	)

	plt.figure(figsize=(12, 8), facecolor="white")
	colors = [CITRINE if c > 0 else MOSS for c in top_corr_df["Correlation"]]

	ax = sns.barplot(
		x="Correlation",
		y="Feature",
		data=top_corr_df,
		palette=colors,
		orient="h"
	)

	ax.set_facecolor("white")

	for bar, value in zip(ax.patches, top_corr_df["Correlation"]):
		x_pos = bar.get_width() / 2 if value > 0 else bar.get_width() - (
				bar.get_width() / 2
		)
		ax.text(
			x=bar.get_x() + x_pos,
			y=bar.get_y() + bar.get_height() / 2,
			s=f"{value:.2f}",
			ha="center",
			va="center",
			color="white",
			fontsize=10,
			fontweight="bold"
		)

	plt.title("Top Correlated Features with Target", fontsize=16)

	plt.axvline(x=0, color="#6B8E23", linestyle="-", alpha=0.3)

	ax.grid(
		which="major",
		axis="x",
		zorder=0,
		color=MOSS,
		linestyle=":",
		dashes=(1, 5),
	)

	plt.tight_layout()
	plt.show()

	return top_corr_df

def dynamic_qq(
	data: pd.DataFrame | np.ndarray,
	variable_names: list[str],
	figsize: tuple = None,
	padding: float = 0.05,
) -> None:
	"""
	Generate Q-Q plots for multiple variables in a dataset.

	Args:
			data (pd.DataFrame | np.ndarray): Dataset containing the variables
			to plot.
			variable_names (list[str]): List of variable names to plot.
			figsize (tuple, optional): Figure size as (width, height). Default
			is calculated based on plots.
			padding (float, optional): Proportion of padding added to
			axis limits. Default is 0.05.

	Returns:
			None: Displays the Q-Q plots.
	"""
	num_plots = len(variable_names)
	num_cols = min(num_plots, 3)
	num_rows = int(np.ceil(num_plots / num_cols))

	if figsize is None:
		figsize = (15, 5 * num_rows)

	fig, axes = plt.subplots(
		num_rows,
		num_cols,
		figsize=figsize,
		squeeze=False,
		dpi=150,
		facecolor=BACKGROUND_COLOR,
	)
	axes = axes.flatten()

	for i, variable_name in enumerate(variable_names):
		if isinstance(data, pd.DataFrame):
			variable_data = data[variable_name].values
		else:
			variable_data = data[:, i]

		sm.qqplot(variable_data, line="45", fit=True, ax=axes[i])

		axes[i].set_facecolor(BACKGROUND_COLOR)
		axes[i].spines["top"].set_visible(False)
		axes[i].spines["right"].set_visible(False)

		axes[i].get_lines()[0].set_markerfacecolor(CITRINE)
		axes[i].get_lines()[0].set_markeredgecolor(CITRINE)
		axes[i].get_lines()[0].set_alpha(0.7)
		axes[i].get_lines()[0].set_zorder(2)

		for child in axes[i].get_children():
			if (
					isinstance(child, plt.matplotlib.lines.Line2D)
					and child != axes[i].get_lines()[0]
			):
				child.set_color(MOSS)
				child.set_linestyle("--")
				child.set_alpha(0.8)
				child.set_zorder(1)

		axes[i].set_title(
			f"Q-Q Plot for {variable_name}",
			fontsize=16,
			fontweight="bold",
			fontfamily="serif",
			pad=15,
		)
		axes[i].set_xlabel(
			axes[i].get_xlabel(),
			fontsize=14,
			fontweight="bold",
			fontfamily="serif",
			labelpad=10,
		)
		axes[i].set_ylabel(
			axes[i].get_ylabel(),
			fontsize=14,
			fontweight="bold",
			fontfamily="serif",
			labelpad=10,
		)

		axes[i].tick_params(axis="x", colors="black")
		axes[i].tick_params(axis="y", colors="black")

		axes[i].grid(False)
		axes[i].grid(
			which="major",
			axis="both",
			zorder=0,
			color=MOSS,
			linestyle=":",
			dashes=(1, 5),
		)

		x_min, x_max = axes[i].get_xlim()
		y_min, y_max = axes[i].get_ylim()
		x_range = x_max - x_min
		y_range = y_max - y_min
		axes[i].set_xlim(x_min - x_range * padding, x_max + x_range * padding)
		axes[i].set_ylim(y_min - y_range * padding, y_max + y_range * padding)

	for i in range(num_plots, len(axes)):
		axes[i].axis("off")

		sns.despine()

	fig.tight_layout()
	plt.show()

