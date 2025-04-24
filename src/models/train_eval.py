import contextlib
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from IPython.display import Markdown, display
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from matplotlib.colors import ListedColormap
from sklearn.base import BaseEstimator
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)


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

def plot_model_evaluation(
	y_true: np.ndarray,
	y_proba: np.ndarray = None,
	classifier=None,
	X: np.ndarray = None,
	figsize: tuple = (9, 4),
	padding: float = 0.05,
) -> None:
	"""
	Plot ROC and Precision-Recall curves for model evaluation.

	Args:
			y_true (np.ndarray): True binary labels.
			y_proba (np.ndarray, optional): Probability estimates of
			the positive class.
			If None, computed using the classifier.
			classifier (object, optional): Fitted classifier with
			`predict_proba` method.
			Required if `y_proba` is None.
			X (np.ndarray, optional): Feature data for predictions.
			Required if `y_proba` is None.
			figsize (tuple, optional): Figure size as (width, height).
			Default is (10, 4).
			padding (float, optional): Proportion of padding added to axis
			limits.
			Default is 0.05.

	Returns:
			None: Displays the plots.
	"""

	if y_proba is None:
		if classifier is None or X is None:
			raise ValueError(
				"If y_proba is None, both classifier and X must be provided"
			)
		y_proba = classifier.predict_proba(X)[:, 1]

	fig, ax = plt.subplots(
		1, 2, figsize=figsize, dpi=150, facecolor=BACKGROUND_COLOR
	)

	fpr, tpr, thresholds = roc_curve(y_true, y_proba)
	roc_auc = roc_auc_score(y_true, y_proba)

	ax[0].plot(
		fpr,
		tpr,
		color=ml_colors[0],
		alpha=0.7,
		linewidth=2,
		zorder=2,
		label=f"AUC: {roc_auc:.2f}",
	)
	ax[0].plot([0, 1], [0, 1], color=MOSS, linestyle="--", alpha=0.8, zorder=1)

	ax[0].set_facecolor(BACKGROUND_COLOR)
	ax[0].spines["top"].set_visible(False)
	ax[0].spines["right"].set_visible(False)

	x_range_roc = fpr.max() - fpr.min()
	y_range_roc = tpr.max() - tpr.min()
	ax[0].set_xlim(
		fpr.min() - x_range_roc * padding, fpr.max() + x_range_roc * padding
	)
	ax[0].set_ylim(
		tpr.min() - y_range_roc * padding, tpr.max() + y_range_roc * padding
	)

	ax[0].grid(
		which="major",
		axis="both",
		zorder=0,
		color=MOSS,
		linestyle=":",
		dashes=(1, 5),
	)

	ax[0].set_xlabel(
		"False Positive Rate",
		fontsize=14,
		fontweight="bold",
		fontfamily="serif",
		labelpad=10,
	)
	ax[0].set_ylabel(
		"True Positive Rate",
		fontsize=14,
		fontweight="bold",
		fontfamily="serif",
		labelpad=10,
	)
	ax[0].set_title(
		"ROC Curve", fontsize=16, fontweight="bold", fontfamily="serif", pad=15
	)

	ax[0].tick_params(axis="x", colors="black")
	ax[0].tick_params(axis="y", colors="black")

	legend = ax[0].legend(loc="lower right", facecolor=BACKGROUND_COLOR)
	legend.get_frame().set_edgecolor(MOSS)

	prec, recall, _ = precision_recall_curve(y_true, y_proba)

	average_precision = average_precision_score(y_true, y_proba)

	ax[1].plot(
		recall, prec, color=ml_colors[0], alpha=0.7, linewidth=2, zorder=2,
		label=f"AP: {average_precision:.2f}"
	)

	legend_pr = ax[1].legend(loc="upper right", facecolor=BACKGROUND_COLOR)
	legend_pr.get_frame().set_edgecolor(MOSS)

	ax[1].set_facecolor(BACKGROUND_COLOR)
	ax[1].spines["top"].set_visible(False)
	ax[1].spines["right"].set_visible(False)

	x_range_pr = recall.max() - recall.min()
	y_range_pr = prec.max() - prec.min()
	ax[1].set_xlim(
		recall.min() - x_range_pr * padding,
		recall.max() + x_range_pr * padding,
	)
	ax[1].set_ylim(
		prec.min() - y_range_pr * padding, prec.max() + y_range_pr * padding
	)

	ax[1].grid(
		which="major",
		axis="both",
		zorder=0,
		color=MOSS,
		linestyle=":",
		dashes=(1, 5),
	)

	ax[1].set_xlabel(
		"Recall",
		fontsize=14,
		fontweight="bold",
		fontfamily="serif",
		labelpad=10,
	)
	ax[1].set_ylabel(
		"Precision",
		fontsize=14,
		fontweight="bold",
		fontfamily="serif",
		labelpad=10,
	)
	ax[1].set_title(
		"Precision-Recall Curve",
		fontsize=16,
		fontweight="bold",
		fontfamily="serif",
		pad=15,
	)

	ax[1].tick_params(axis="x", colors="black")
	ax[1].tick_params(axis="y", colors="black")

	sns.despine()

	fig.tight_layout()
	plt.show()


def shap_summary(
	model: BaseEstimator,
	X: np.ndarray,
	feature_names: List[str] = None,
	model_name: Optional[str] = "Model",
	padding: float = 0.05,
) -> None:
	"""
	Generate a SHAP summary plot with consistent evaluation style.

	Args:
			model (BaseEstimator): Trained model compatible with SHAP
			X (np.ndarray): Feature matrix for SHAP computation
			feature_names (List[str]): Feature names for labeling
			model_name (str, optional): Model name for title. Default "Model"
			figsize (tuple): Figure dimensions. Default (9, 4)
			padding (float): Axis padding proportion. Default 0.05

	Returns:
			None: Displays styled SHAP summary plot
	"""
	fig = plt.figure(facecolor=BACKGROUND_COLOR, dpi=90)
	ax = fig.gca()

	explainer = shap.Explainer(model,
	                           feature_perturbation="tree_path_dependent")
	shap_values = explainer(X)
	shap.summary_plot(
		shap_values, X, feature_names=feature_names, show=False, cmap=cmap
	)

	ax.set_facecolor(BACKGROUND_COLOR)
	ax.set_xlabel(
		"Feature Impact",
		fontsize=14,
		fontweight="bold",
		fontfamily="serif",
		labelpad=10,
	)
	ax.set_title(
		f"SHAP Summary - {model_name}",
		fontsize=16,
		fontweight="bold",
		fontfamily="serif",
		pad=15,
	)

	x_min, x_max = ax.get_xlim()
	x_range = x_max - x_min
	ax.set_xlim(x_min - x_range * padding, x_max + x_range * padding)

	ax.tick_params(axis="both", which="major", labelsize=8, colors="black")
	for spine in ax.spines.values():
		spine.set_color(MOSS)
		spine.set_linewidth(0.8)

	ax.grid(
		True,
		which="major",
		axis="both",
		zorder=0,
		color=MOSS,
		linestyle=":",
		dashes=(1, 5),
		alpha=0.7,
	)

	cbar = fig.axes[-1]
	cbar.set_ylabel(
		"Feature Value",
		fontsize=10,
		fontfamily="serif",
		fontweight="bold",
		color=MOSS,
		labelpad=10,
	)
	cbar.tick_params(labelsize=10, colors=MOSS)
	cbar.spines[:].set_color(MOSS)
	cbar.set_facecolor(BACKGROUND_COLOR)

	plt.tight_layout()
	plt.show()


def feature_importances(

	model: BaseEstimator,
	df: pd.DataFrame,
	top_n: int = 10,
	figsize: Tuple[int, int] = (9, 8),
	model_name: str = "Model",
	padding: float = 0.05,
	importance_threshold: float = 0,
	xmin: float = None,
	xmax: float = None,
) -> Optional[List[str]]:
	"""
	Plot styled top feature importances for a model.

	Args:
		model (BaseEstimator): Fitted classifier with feature_importances_.
		df (pd.DataFrame): DataFrame containing the features.
		top_n (int): Number of top features to show.
		figsize (Tuple[int, int]): Plot size (width, height) in inches.
		model_name (str): Model name for plot title.
		padding (float): Proportional padding for axis limits.
		importance_threshold (float): Optional threshold for feature importance.
		xmin (float): Minimum value for x-axis (if None, auto-calculated).
		xmax (float): Maximum value for x-axis (if None, auto-calculated).

	Returns:
		Optional[List[str]]: List of features below threshold if specified
	"""
	feature_names = df.columns.tolist()
	importances = model.feature_importances_

	importances_df = pd.DataFrame(
		{"Feature": feature_names, "Importance": importances}
	).sort_values(by="Importance", ascending=False)

	top_features = importances_df.head(top_n)

	fig, ax = plt.subplots(
		figsize=figsize, facecolor=BACKGROUND_COLOR, dpi=150
	)

	sns.barplot(
		x="Importance",
		y="Feature",
		data=top_features,
		palette=[CITRINE],
		ax=ax,
	)
	ax.set_title(
		f"{model_name} - Top {top_n} Features",
		fontsize=14,
		fontweight="bold",
		fontfamily="serif",
	)
	ax.set_xlabel(
		"Importance", fontsize=12, fontweight="bold", fontfamily="serif"
	)
	ax.set_ylabel("", fontsize=12, fontweight="bold", fontfamily="serif")
	ax.set_facecolor(BACKGROUND_COLOR)

	if xmin is not None or xmax is not None:
		x_min_top, x_max_top = ax.get_xlim()
		ax.set_xlim(
			xmin if xmin is not None else x_min_top,
			xmax if xmax is not None else x_max_top
		)
	else:
		x_min_top, x_max_top = ax.get_xlim()
		x_range_top = x_max_top - x_min_top
		ax.set_xlim(
			x_min_top - x_range_top * padding,
			x_max_top + x_range_top * padding
		)

	ax.tick_params(axis="x", colors="black")
	ax.tick_params(axis="y", colors="black", labelsize=8)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.grid(axis="x", color=MOSS, linestyle="--")

	fig.tight_layout()
	plt.show()

	if importance_threshold is not None:
		low_importance_features = importances_df[
			importances_df["Importance"] < importance_threshold
			]["Feature"].tolist()

		display(Markdown(
			f"**{len(low_importance_features)} features** were found below "
			f"threshold {importance_threshold}"
		))

		return low_importance_features

	return None


def dynamic_confusion_matrix(
	models=None, X=None, y=None, display_labels=None,
	model_names=None, custom_matrices=None
):
	"""Plot confusion matrices for models or custom matrices.

	Parameters
	----------
	models : List, optional
		Trained ML models. Not needed if custom_matrices provided.
	X : DataFrame, optional
		Feature matrix. Not needed if custom_matrices provided.
	y : Series, optional
		Target variable. Not needed if custom_matrices provided.
	display_labels : List[str], optional
		Class labels for confusion matrix.
	model_names : List[str], optional
		Names for each model/matrix. Uses class names if not provided.
	custom_matrices : List, optional
		Pre-computed confusion matrices to display.
	"""
	if custom_matrices is not None:
		num_matrices = len(custom_matrices)
		if model_names is None:
			model_names = [f"Matrix {i + 1}" for i in range(num_matrices)]
	elif models is not None:
		num_matrices = len(models)
		if model_names is None:
			model_names = [model.__class__.__name__ for model in models]
	else:
		raise ValueError("Either models or custom_matrices must be provided")

	if len(model_names) != num_matrices:
		raise ValueError(
			f"Expected {num_matrices} model names, got {len(model_names)}"
		)

	fig, axes = plt.subplots(1, num_matrices,
	                         figsize=(5 * num_matrices, 5))
	if num_matrices == 1:
		axes = [axes]

	if display_labels is None:
		if custom_matrices is not None and len(custom_matrices[0]) == 2:
			display_labels = ['0', '1']
		else:
			raise ValueError("display_labels must be provided")

	for i in range(num_matrices):
		if custom_matrices is not None:
			cm = custom_matrices[i]
		else:
			model = models[i]
			y_pred = model.predict(X)
			cm = confusion_matrix(y, y_pred)

		cm_percentage = (
				cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
		)

		disp = ConfusionMatrixDisplay(
			confusion_matrix=cm, display_labels=display_labels
		)
		disp.plot(ax=axes[i], cmap=cmap, values_format='d')
		axes[i].set_title(
			model_names[i], fontsize=14, fontweight="bold"
		)

		for j, text in enumerate(disp.text_.ravel()):
			text.set_fontsize(18)
			text.set_fontweight("bold")
			row, col = divmod(j, len(display_labels))
			percentage = cm_percentage[row, col]
			color = text.get_color()
			axes[i].text(
				col, row + 0.15, f'({percentage:.1f}%)',
				ha='center', va='center', fontsize=14, color=color
			)

	plt.tight_layout()
	plt.show()


def fit_lgbm_model(
	model: LGBMClassifier,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: pd.DataFrame,
	y_val: pd.Series,
	early_stopping_rounds: int = 30,
	metric: str = "auc",
) -> Tuple[LGBMClassifier, np.ndarray]:
	"""Fit LightGBM model and evaluate performance metrics.

	Args:
		model: LightGBM classifier to train
		X_train: Training feature matrix
		y_train: Training target vector
		X_val: Validation feature matrix
		y_val: Validation target vector
		early_stopping_rounds: Rounds for early stopping
		metric: Evaluation metric for training

	Returns:
		Tuple of fitted model and validation predictions
	"""
	cat_features = X_train.select_dtypes(
		include=["category", "object"]
	).columns.tolist()

	model.fit(
		X_train,
		y_train,
		categorical_feature=cat_features,
		eval_set=[(X_train, y_train), (X_val, y_val)],
		eval_names=['train', 'valid'],
		eval_metric=metric,
		callbacks=[
			log_evaluation(period=100),
			early_stopping(early_stopping_rounds, verbose=False)
		]
	)

	train_preds = model.predict_proba(X_train)[:, 1]
	val_preds = model.predict_proba(X_val)[:, 1]

	pr_auc_train = average_precision_score(y_train, train_preds)
	pr_auc_val = average_precision_score(y_val, val_preds)
	roc_auc_train = roc_auc_score(y_train, train_preds)
	roc_auc_val = roc_auc_score(y_val, val_preds)

	display(Markdown("**LightGBM Model Performance**"))

	display(Markdown("**PR AUC Scores**"))
	display(Markdown(f"- **Train**: {pr_auc_train:.4f}"))
	display(Markdown(f"- **Validation**: {pr_auc_val:.4f}"))

	display(Markdown("**ROC AUC Scores**"))
	display(Markdown(f"- **Train**: {roc_auc_train:.4f}"))
	display(Markdown(f"- **Validation**: {roc_auc_val:.4f}"))

	pr_diff = pr_auc_train - pr_auc_val
	roc_diff = roc_auc_train - roc_auc_val

	display(Markdown("**Overfitting Analysis**"))
	display(Markdown(f"- **PR AUC Difference**: {pr_diff:.4f}"))
	display(Markdown(f"- **ROC AUC Difference**: {roc_diff:.4f}"))

	if pr_diff > 0.05 or roc_diff > 0.05:
		display(Markdown("⚠️ **Warning**: Potential overfitting detected"))

	return model, val_preds


def lgbm_objective(trial: optuna.Trial,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: pd.DataFrame,
                   y_val: pd.Series) -> float:
	"""Objective function for credit risk model hyperparameter optimization."""
	num_leaves = trial.suggest_int("num_leaves", 16, 256)
	min_child_samples = trial.suggest_int("min_child_samples", 10, 100)
	min_child_weight = trial.suggest_float("min_child_weight", 1e-5, 1.0)
	colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 0.8)
	subsample = trial.suggest_float("subsample", 0.6, 1.0)
	max_depth = trial.suggest_int("max_depth", 3, 12)
	reg_lambda = trial.suggest_float("reg_lambda", 0.005, 10, log=True)
	n_estimators = trial.suggest_int("n_estimators", 500, 10000)
	reg_alpha = trial.suggest_float("reg_alpha", 0.005, 10, log=True)
	learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
	feature_fraction = trial.suggest_float("feature_fraction", 0.6, 1.0)
	bagging_freq = trial.suggest_int("bagging_freq", 1, 10)
	scale_pos_weight = trial.suggest_float("scale_pos_weight", 1, 100)

	model = LGBMClassifier(
		n_estimators=n_estimators,
		learning_rate=learning_rate,
		num_leaves=num_leaves,
		min_child_samples=min_child_samples,
		min_child_weight=min_child_weight,
		colsample_bytree=colsample_bytree,
		subsample=subsample,
		max_depth=max_depth,
		reg_lambda=reg_lambda,
		reg_alpha=reg_alpha,
		feature_fraction=feature_fraction,
		bagging_freq=bagging_freq,
		n_jobs=-1,
		verbose=-1,
		scale_pos_weight=scale_pos_weight,
	)

	cat_features = X_train.select_dtypes(
		include=["category", "object"]
	).columns.tolist()

	pruning_callback = optuna.integration.LightGBMPruningCallback(
		trial,
		metric="average_precision",
		valid_name="valid_0"
	)

	with open(os.devnull, "w") as fnull:
		with contextlib.redirect_stdout(fnull):
			model.fit(
				X_train,
				y_train,
				categorical_feature=cat_features if cat_features else "auto",
				eval_set=[(X_val, y_val)],
				eval_metric="average_precision",
				callbacks=[
					log_evaluation(period=100),
					early_stopping(50),
					pruning_callback
				]
			)

	y_pred = model.predict_proba(X_val)[:, 1]
	pr_auc = average_precision_score(y_val, y_pred)

	return pr_auc


def optimize_lgbm_model(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series,
                        n_trials: int = 50
                        ) -> Tuple[optuna.study.Study, Dict[str, Any]]:

	"""Optimize LightGBM model hyperparameters using Optuna.

	text
	Args:
		X_train: Training features
		y_train: Training target
		X_val: Validation features
		y_val: Validation target
		n_trials: Number of optimization trials

	Returns:
		Tuple of Optuna study and best parameters
	"""

	objective = partial(
		lgbm_objective,
		X_train=X_train,
		y_train=y_train,
		X_val=X_val,
		y_val=y_val
	)

	optuna.logging.set_verbosity(optuna.logging.ERROR)
	study = optuna.create_study(
		direction="maximize",
		pruner=optuna.pruners.HyperbandPruner()
	)
	study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

	best_trial = study.best_trial
	print(f"Best PR-AUC: {best_trial.value:.4f}")
	print("Best parameters:")
	for key, value in best_trial.params.items():
		print(f"    {key}: {value}")

	return study, best_trial.params


def fit_optuna_lgbm(study: optuna.Study,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: pd.DataFrame,
                    y_val: pd.Series) -> LGBMClassifier:
	"""
	Fit a tuned LightGBM model using the best parameters from an Optuna study.

	Args:
		study: Completed Optuna study
		X_train: Training features
		y_train: Training target
		X_val: Validation features
		y_val: Validation target

	Returns:
		Fitted LGBMClassifier
	"""
	best_params = study.best_trial.params
	lgb_tuned = LGBMClassifier(verbose=-1, **best_params)

	cat_features = X_train.select_dtypes(
		include=["category", "object"]
	).columns.tolist()

	with open(os.devnull, "w") as fnull:
		with contextlib.redirect_stdout(fnull):
			lgb_tuned.fit(
				X_train,
				y_train,
				categorical_feature=cat_features if cat_features else "auto",
				eval_set=[(X_val, y_val)],
				eval_metric="average_precision",
				callbacks=[log_evaluation(period=100), early_stopping(50)],
			)

	return lgb_tuned