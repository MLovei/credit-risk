import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

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


def graph_all(
	df: pd.DataFrame,
	column_names: Optional[List[str]] = None,
	titles: Optional[List[str]] = None,
	max_cols: int = 3,
	figsize: Tuple[int, int] = (15, 12),
	plot_type: str = "histplot",
	bins: int = 30,
	hist_color: str = APPLE,
	line_color: str = MOSS,
) -> None:
	"""
	Generates a grid of plots (histograms or boxplots) using Seaborn (sns)
	with custom styling, dynamically adjusting the number of rows and columns
	based on the number of features.

	Args:
			df: pandas DataFrame containing the data.
			column_names: A list of column names to plot.
									  If None, all numeric columns are used for
									  histograms,
									  and all object/categorical columns are
									  used for boxplots.
			titles: A list of titles for each subplot (optional).
							If not provided, column names are used.
			max_cols: Maximum number of columns per row in
			the subplot grid (default: 3).
			figsize: Figure size (width, height) in inches (default: (15, 12)).
			plot_type: Type of plot: "histplot" (default) or "boxplot".
			bins: The number of bins in the histograms (default: 30).
			hist_color: The color of the histogram bars (default: APPLE).
			line_color: The color of the grid lines and outline (default: MOSS).

	Returns:
			None. Displays the plot.
	"""

	if column_names is None:
		if plot_type == "histplot":
			column_names = df.select_dtypes(include=np.number).columns.tolist()
		elif plot_type == "boxplot":
			column_names = df.select_dtypes(
				include=["object", "category"]
			).columns.tolist()
		else:
			raise ValueError(
				"Invalid plot_type. Choose 'histplot' or" " 'boxplot'."
			)

	num_plots = len(column_names)
	if num_plots == 0:
		return

	num_rows = math.ceil(num_plots / max_cols)

	fig, axes = plt.subplots(
		nrows=num_rows,
		ncols=max_cols,
		figsize=figsize,
		facecolor=BACKGROUND_COLOR,
	)

	if num_plots == 1:
		axes = [axes]
	elif num_rows == 1:
		axes = axes.reshape(1, -1)[0]
	else:
		axes = axes.flatten()

	if titles is None:
		titles = [col.replace("_",
		                      " ").title() for col in column_names]

	for i, (col, title) in enumerate(zip(column_names, titles)):
		ax = axes[i]
		ax.set_facecolor(BACKGROUND_COLOR)
		for s in ["top", "right", "left"]:
			ax.spines[s].set_visible(False)

		data = df[col].dropna()

		if plot_type == "histplot":
			sns.histplot(
				data,
				bins=bins,
				ax=ax,
				color=hist_color,
				edgecolor=line_color,
				linewidth=1.5,
				kde=False,
				alpha=0.8,
			)
			ax.grid(
				which="major",
				axis="y",
				zorder=0,
				color=line_color,
				linestyle=":",
				dashes=(1, 5),
			)
			ax.set_xlabel(
				col.replace("_", " ").title(), fontsize=12, fontweight="bold"
			)
			ax.set_title(
				title, fontsize=14, fontweight="bold", fontfamily="serif"
			)
			ax.tick_params(axis="y", labelsize=10)

		elif plot_type == "boxplot":
			sns.boxplot(
				y=data,
				ax=ax,
				color=hist_color,
				width=0.6,
				**{
					"boxprops"    : {"edgecolor": line_color, "linewidth": 1.5},
					"medianprops" : {"color": line_color, "linewidth": 1.5},
					"whiskerprops": {"color": line_color, "linewidth": 1.5},
					"capprops"    : {"color": line_color, "linewidth": 1.5},
					"flierprops"  : {
						"markerfacecolor": hist_color,
						"markeredgecolor": line_color,
						"markersize"     : 8,
					},
				},
			)
			ax.set_ylabel(
				col.replace("_", " ").title(), fontsize=12, fontweight="bold"
			)
			ax.set_title(
				title, fontsize=14, fontweight="bold", fontfamily="serif"
			)
			ax.tick_params(axis="y", labelsize=10)

		else:
			raise ValueError(
				"Invalid plot_type." " Choose 'histplot' or 'boxplot'."
			)

	if num_plots < num_rows * max_cols:
		for i in range(num_plots, num_rows * max_cols):
			axes[i].set_axis_off()

	plt.tight_layout()
	plt.show()


def graph_by_group(
	df: pd.DataFrame,
	group_variable: str,
	target_variables: list,
	legend_x: float = 0.95,
	legend_y: float = 0.9,
) -> None:
	"""
	Generates a grid of plots showing the distribution of each target variable
	based on the categories of a group variable.

	Args:
		df (pd.DataFrame): DataFrame containing the data.
		group_variable (str): The name of the grouping variable.
		target_variables (list): A list of names of the target variables to
			visualize.
		legend_x (float): X-coordinate for the legend's position
			(default: 0.95).
		legend_y (float): Y-coordinate for the legend's position
			(default: 0.9).

	Returns:
		None. Displays the plot.
	"""
	num_targets = len(target_variables)
	num_cols = 1
	num_rows = (num_targets + 1) // num_cols

	fig = plt.figure(
		figsize=(12, 5 * num_rows), dpi=150, facecolor=BACKGROUND_COLOR
	)
	gs = fig.add_gridspec(num_rows, num_cols)
	gs.update(wspace=0.1, hspace=0.4)

	categories = df[group_variable].unique()

	predefined_colors = [MOSS, CITRINE, APPLE, CREAM, NAPLES]

	if len(categories) <= 5:
		colors_list = predefined_colors
	else:
		raise ValueError(
			f"Not enough colors defined in the palette for variable "
			f"'{group_variable}'. Please add at least "
			f"{len(categories) - len(predefined_colors)} more colors."
		)

	colors = {
		category: colors_list[i % len(colors_list)]
		for i, category in enumerate(categories)
	}

	for i, target_variable in enumerate(target_variables):
		row = i // num_cols
		col = i % num_cols
		ax = fig.add_subplot(gs[row, col])

		ax.set_facecolor(BACKGROUND_COLOR)

		legend_elements = []

		for idx, category in enumerate(categories):
			subset = df[df[group_variable] == category]

			if pd.api.types.is_numeric_dtype(df[target_variable]):
				sns.kdeplot(
					subset[target_variable],
					ax=ax,
					color=colors[category],
					fill=True,
					linewidth=2,
					ec=MOSS,
					alpha=0.7,
				)

				median_value = subset[target_variable].median()
				ax.axvline(
					median_value,
					color=colors[category],
					linestyle="--",
				)

				if idx % 2 == 0:
					y_position = ax.get_ylim()[1] * 0.9
				else:
					y_position = ax.get_ylim()[0] + ax.get_ylim()[1] * 0.1

				ax.text(
					median_value,
					y_position,
					f"{median_value:.2f}",
					color="white",
					fontsize=10,
					fontweight="bold",
					ha="center",
					va="center",
					bbox=dict(facecolor=colors[category], edgecolor=None),
				)
			else:
				counts = subset[target_variable].value_counts()
				total_count = counts.sum()
				percentages = (counts / total_count) * 100
				bars = ax.bar(
					counts.index.astype(str),
					counts.values,
					color=colors[category],
					edgecolor=MOSS,
					linewidth=1,
				)
				for bar, percentage in zip(bars, percentages):
					height = bar.get_height()
					ax.text(
						bar.get_x() + bar.get_width() / 2,
						height,
						f"{percentage:.1f}%",
						ha="center",
						va="bottom",
						fontsize=9,
					)

			ax.autoscale_view()

			legend_elements.append(
				Line2D(
					[0], [0], color=colors[category], lw=2,
					label=f"{group_variable}={category}"
				)
			)
			legend_elements.append(
				Line2D(
					[0], [0], color=colors[category], linestyle="--", lw=1.5,
					label=f"Median ({group_variable}={category})"
				)
			)

		ax.grid(
			which="major",
			axis="x",
			zorder=0,
			color=MOSS,
			linestyle=":",
			dashes=(1, 5),
		)
		ax.set_title(
			f"'{target_variable}' Distribution by '{group_variable}' feature",
			fontsize=14,
			fontweight="bold",
			fontfamily="serif",
		)

		fig.legend(
			handles=legend_elements,
			loc="upper right",
			bbox_to_anchor=(legend_x, legend_y),
			fontsize=10,
		)

	plt.tight_layout()
	plt.show()


def scatterplot(
	df: pd.DataFrame,
	x_variable: str,
	y_variable: str,
	x_min_clip: float = None,
	x_max_clip: float = None,
	y_min_clip: float = None,
	y_max_clip: float = None,
	padding: float = 0.05,
) -> None:
	"""
	Generates a scatterplot of two variables.

	Args:
	  df: pandas DataFrame containing the data.
	  x_variable: The name of the variable for the x-axis.
	  y_variable: The name of the variable for the y-axis.
	  x_min_clip: The minimum value for the x-axis.
	  x_max_clip: The maximum value for the x-axis.
	  y_min_clip: The minimum value for the y-axis.
	  y_max_clip: The maximum value for the y-axis.
	  padding: The proportion of padding to add to the axes limits.

	Returns:
	  None. Displays the plot.
	"""

	fig, ax = plt.subplots(
		figsize=(12, 8), dpi=150, facecolor=BACKGROUND_COLOR
	)

	plt.gca().spines["top"].set_visible(False)
	plt.gca().spines["right"].set_visible(False)

	plt.scatter(
		df[x_variable], df[y_variable], color=CITRINE, alpha=0.7, zorder=2
	)

	if x_min_clip is None:
		x_min_clip = df[x_variable].min()
		x_range = df[x_variable].max() - x_min_clip
		x_min_clip -= x_range * padding
	if x_max_clip is None:
		x_max_clip = df[x_variable].max()
		x_range = x_max_clip - df[x_variable].min()
		x_max_clip += x_range * padding
	if y_min_clip is None:
		y_min_clip = df[y_variable].min()
		y_range = df[y_variable].max() - y_min_clip
		y_min_clip -= y_range * padding
	if y_max_clip is None:
		y_max_clip = df[y_variable].max()
		y_range = y_max_clip - df[y_variable].min()
		y_max_clip += y_range * padding

	plt.xlim(x_min_clip, x_max_clip)
	plt.ylim(y_min_clip, y_max_clip)

	ax.set_facecolor(BACKGROUND_COLOR)
	ax.grid(False)
	ax.grid(
		which="major",
		axis="both",
		zorder=0,
		color=MOSS,
		linestyle=":",
		dashes=(1, 5),
	)
	ax.set_xlabel(
		x_variable, fontsize=14, fontweight="bold", fontfamily="serif"
	)
	ax.set_ylabel(
		y_variable, fontsize=14, fontweight="bold", fontfamily="serif"
	)
	ax.set_title(
		f"Scatterplot of '{y_variable}' vs. '{x_variable}'",
		fontsize=16,
		fontweight="bold",
		fontfamily="serif",
	)
	fig.tight_layout()
	plt.show()


def stacked_barcharts_by_group(
	df: pd.DataFrame,
	group_variable: str,
	target_variables: list,
	transpose: bool = False,
	legend_x: float = 0.95,
	legend_y: float = 0.95,
) -> None:
	"""
	Generates a grid of *normalized* stacked bar charts.
	"""

	num_targets = len(target_variables)
	num_cols = 2
	num_rows = (num_targets + 1) // num_cols

	fig = plt.figure(
		figsize=(12, 6 * num_rows), dpi=150, facecolor=BACKGROUND_COLOR
	)
	gs = fig.add_gridspec(num_rows, num_cols)
	gs.update(wspace=0.3, hspace=0.3)
	legend_data = None

	for i, target_variable in enumerate(target_variables):
		row = i // num_cols
		col = i % num_cols
		ax = fig.add_subplot(gs[row, col])

		ax.set_facecolor(BACKGROUND_COLOR)
		ax.tick_params(axis="y", left=False)
		ax.get_yaxis().set_visible(False)
		for s in ["top", "right", "left"]:
			ax.spines[s].set_visible(False)

		df_dropped = df.dropna(subset=[target_variable])

		if not transpose:
			grouped_data = (
				df_dropped.groupby([group_variable, target_variable])
				.size()
				.unstack(fill_value=0)
			)
			group_totals = grouped_data.sum(axis=1)
			percentages = grouped_data.div(group_totals, axis=0) * 100
			categories = df_dropped[target_variable].unique()

			bottom = np.zeros(len(grouped_data))
			for j, category in enumerate(categories):
				values = percentages[category].values
				bars = ax.bar(
					grouped_data.index.astype(str),
					values,
					bottom=bottom,
					label=category,
					color=full_pallet[j],
					edgecolor=MOSS,
					linewidth=2,
				)

				for bar, percentage in zip(bars, values):
					height = bar.get_height()
					if height > 0:
						label_position = bar.get_y() + height / 2
						ax.text(
							bar.get_x() + bar.get_width() / 2,
							label_position,
							f"{percentage:.1f}%",
							ha="center",
							va="center",
							fontsize=9,
							color="black",
						)

				bottom += values

			ax.set_ylim(0, 100)

			ax.set_title(
				f"'{target_variable}' Distribution by "
				f"'{group_variable}'",
				fontsize=14,
				fontweight="bold",
				fontfamily="serif",
				y=1.05,
			)

		else:
			grouped_data = (
				df_dropped.groupby([target_variable, group_variable])
				.size()
				.unstack(fill_value=0)
			)
			group_totals = grouped_data.sum(axis=1)
			percentages = grouped_data.div(group_totals, axis=0) * 100
			categories = df_dropped[group_variable].unique()

			bottom = np.zeros(len(grouped_data))
			for j, category in enumerate(categories):
				values = percentages[category].values
				bars = ax.bar(
					grouped_data.index.astype(str),
					values,
					bottom=bottom,
					label=category,
					color=full_pallet[j],
					edgecolor=MOSS,
					linewidth=2,
				)

				for bar, percentage in zip(bars, values):
					height = bar.get_height()
					if height > 0:
						label_position = bar.get_y() + height / 2
						ax.text(
							bar.get_x() + bar.get_width() / 2,
							label_position,
							f"{percentage:.1f}%",
							ha="center",
							va="center",
							fontsize=9,
							color="black",
						)

				bottom += values

			ax.set_ylim(0, 100)

			ax.set_title(
				f"'{group_variable}' Distribution by"
				f" '{target_variable}'",
				fontsize=14,
				fontweight="bold",
				fontfamily="serif",
				y=1.05,
			)

		ax.grid(
			which="major",
			axis="x",
			zorder=0,
			color=MOSS,
			linestyle=":",
			dashes=(1, 5),
		)

		if i == 0:
			legend_data = {
				"categories"     : categories,
				"colors"         : full_pallet,
				"transpose"      : transpose,
				"group_variable" : group_variable,
				"target_variable": target_variable,
			}

	legend_elements = []
	if legend_data is not None:
		for j, category in enumerate(legend_data["categories"]):
			if not legend_data["transpose"]:
				label = f"{legend_data["target_variable"]} = {category}"
			else:
				label = f"{legend_data["group_variable"]} = {category}"

			legend_elements.append(
				Line2D(
					[0], [0], color=legend_data["colors"][j],
					lw=2, label=label
				)
			)

	fig.legend(
		handles=legend_elements,
		loc="upper right",
		bbox_to_anchor=(legend_x, legend_y),
		fontsize=10,
	)

	plt.tight_layout()
	plt.show()


def barcharts_by_group(
	df: pd.DataFrame,
	target_variables: list,
	show_yaxis: bool = True,
	show_legend: bool = False,
	legend_labels: list = None,
	orientation: str = "vertical",
	padding: float = 0.05,
	wspace: float = 0.4,
	hspace: float = 0.6,
	use_single_color: bool = True,
) -> None:
	"""Generates bar plots for target variables in a grid layout.

	Args:
		df (pd.DataFrame): DataFrame containing the data.
		target_variables (list): Names of the target variable columns.
		show_yaxis (bool): Whether to show the y-axis.
		show_legend (bool): Whether to show a legend.
		legend_labels (list): Labels for the legend.
		orientation (str): Bar orientation ("vertical" or "horizontal").
		padding (float): Padding for axis limits.
		wspace (float): Horizontal space between plots.
		hspace (float): Vertical space between plots.
		use_single_color (bool): Whether to use a single color
			for all bars. Default is True.

	Returns:
		None. Displays the plot.
	"""
	num_targets = len(target_variables)
	num_cols = 2
	num_rows = (num_targets + 1) // num_cols

	def format_percentage(percentage):
		"""Formats percentage values for display on bar charts."""
		if percentage < 0.01:
			return "<0.01%"
		elif percentage.is_integer():
			return f"{percentage:.0f}%"
		elif (percentage * 10).is_integer():
			return f"{percentage:.1f}%"
		else:
			return f"{percentage:.2f}%"

	fig = plt.figure(
		figsize=(6 * num_cols, 4 * num_rows),
		dpi=150,
		facecolor=BACKGROUND_COLOR,
	)
	gs = fig.add_gridspec(num_rows, num_cols)
	gs.update(wspace=wspace, hspace=hspace)

	for i, target_variable in enumerate(target_variables):
		row = i // num_cols
		col = i % num_cols
		ax = fig.add_subplot(gs[row, col])

		ax.set_facecolor(BACKGROUND_COLOR)

		if not show_yaxis:
			ax.tick_params(axis="y", left=False)
			ax.get_yaxis().set_visible(False)
			for spine in ["top", "right", "left"]:
				ax.spines[spine].set_visible(False)
		else:
			ax.tick_params(axis="y", left=True)
			ax.get_yaxis().set_visible(True)
			for spine in ["top", "right"]:
				ax.spines[spine].set_visible(False)

			ax.spines["left"].set_visible(True)
			ax.spines["left"].set_color(MOSS)

		counts = df[target_variable].value_counts()
		counts = counts.sort_values(ascending=False)
		percentages = (counts / len(df) * 100).round(4)

		if use_single_color:
			colors = [CITRINE] * len(counts)
		else:
			colors = [APPLE, CITRINE, CREAM, NAPLES,
			          MOSS, OLIVE, HARVEST, PEAR,
			          ]
			if len(counts) > len(colors):
				raise ValueError(
					"Not enough colors defined in the palette for variable "
					f"'{target_variable}'. "
					"Add more colors to `colors` list."
				)

		if orientation == "vertical":
			bars = ax.bar(
				counts.index.astype(str),
				counts.values,
				color=colors[: len(counts)],
				edgecolor=MOSS,
				linewidth=2,
			)
			for bar, percentage in zip(bars, percentages):
				height = bar.get_height()
				formatted_pct = format_percentage(percentage)
				ax.text(
					bar.get_x() + bar.get_width() / 2,
					height + height * padding,
					formatted_pct,
					ha="center",
					va="bottom",
					fontsize=10,
				)
			ax.set_xticks(range(len(counts)))
			ax.set_xticklabels(counts.index.astype(str))
			if show_yaxis:
				ax.set_ylabel("Count", fontsize=12)

			x_min_top, x_max_top = ax.get_xlim()
			x_range_top = x_max_top - x_min_top
			ax.set_xlim(
				x_min_top - x_range_top * padding,
				x_max_top + x_range_top * padding,
			)

		elif orientation == "horizontal":
			bars = ax.barh(
				counts.index.astype(str),
				counts.values,
				color=colors[: len(counts)],
				edgecolor=MOSS,
				linewidth=2,
			)
			for bar, percentage in zip(bars, percentages):
				width = bar.get_width()
				formatted_pct = format_percentage(percentage)
				ax.text(
					width + width * padding,
					bar.get_y() + bar.get_height() / 2,
					formatted_pct,
					ha="left",
					va="center",
					fontsize=10,
				)
			ax.set_yticks(range(len(counts)))
			ax.set_yticklabels(counts.index.astype(str))
			if show_yaxis:
				ax.set_xlabel("Count", fontsize=12)

			y_min_bottom, y_max_bottom = ax.get_ylim()
			y_range_bottom = y_max_bottom - y_min_bottom
			ax.set_ylim(
				y_min_bottom - y_range_bottom * padding,
				y_max_bottom + y_range_bottom * padding,
			)

		if show_legend and legend_labels:
			legend_handles = [
				plt.Rectangle((0, 0), 1, 1, color=colors[i])
				for i in range(len(legend_labels))
			]
			ax.legend(
				legend_handles,
				legend_labels,
				loc="upper right",
				frameon=True,
				facecolor="white",
				framealpha=0.9,
			)

		ax.set_title(
			f"'{target_variable.replace('_', ' ').title()}'"
			f" Feature Distribution",
			fontsize=14,
			fontweight="bold",
			fontfamily="serif",
		)

	plt.tight_layout()
	plt.show()








