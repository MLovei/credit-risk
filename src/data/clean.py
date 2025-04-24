from typing import List, Tuple

import numpy as np
import pandas as pd


def optimize_dtypes(df):
	"""
	Downcasts numerical columns to the smallest possible data types
	(integers and floats)
	and converts object columns to categorical, which reduces memory usage.

	Parameters:
	df (pd.DataFrame): The DataFrame to optimize.

	Returns:
	pd.DataFrame: A new DataFrame with optimized memory usage.
	"""
	result = df.copy()

	int_columns = result.select_dtypes(include=["int"])
	if not int_columns.empty:
		result[int_columns.columns] = int_columns.apply(
			pd.to_numeric, downcast="integer"
		)

	float_columns = result.select_dtypes(include=["float"])
	if not float_columns.empty:
		result[float_columns.columns] = float_columns.apply(
			pd.to_numeric, downcast="float"
		)

	obj_columns = result.select_dtypes(include=["object"])
	if not obj_columns.empty:
		for col in obj_columns.columns:
			if obj_columns[col].nunique() < 0.5 * len(obj_columns[col]):
				result[col] = obj_columns[col].astype("category")

	return result


def clean_creditrisk(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Cleans the credit risk DataFrame by handling missing values, outliers,
	and creating derived features.

	Args:
		df: Raw DataFrame from the credit risk dataset

	Returns:
		Cleaned DataFrame ready for feature engineering and modeling
	"""

	df = df.copy()

	replacements = {
		"XNA": np.nan,
		"XAP": np.nan,
	}
	df.replace(replacements, inplace=True)

	df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
	df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan, inplace=True)

	df["AGE_YEARS"] = abs(
		df["DAYS_BIRTH"]) / 365.25
	df["YEARS_EMPLOYED"] = abs(df["DAYS_EMPLOYED"]) / 365.25

	df["EMPLOYMENT_RATIO"] = df["YEARS_EMPLOYED"] / df[
		"AGE_YEARS"]

	df["EMPLOYMENT_RATIO"] = df["EMPLOYMENT_RATIO"].clip(upper=1)

	house_cols_drop = [col for col in df.columns if
	                   col.endswith(("MEDI", "MODE"))]
	df.drop(columns=house_cols_drop, inplace=True)

	df.drop(columns=["DAYS_EMPLOYED", "DAYS_BIRTH"], inplace=True)

	df["NAME_EDUCATION_TYPE"].replace(
		"Secondary / secondary special", "Secondary", inplace=True
	)

	if 'TARGET' in df.columns:
		df["TARGET"] = df["TARGET"].astype(int)

	return df


def preprocess_dataframe(
	df: pd.DataFrame, categorical_cols: List[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
	"""
	Preprocesses a DataFrame by encoding categorical columns,
	cleaning column names, and separating features and target.

	Args:
		df (pd.DataFrame): Input DataFrame.
		categorical_cols (List[str]): List of categorical column names.
		target_col (str): Name of the target column.

	Returns:
		Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).
	"""
	if target_col not in df.columns:
		raise ValueError(
			f"Target column '{target_col}' not found in DataFrame.")

	if not all(col in df.columns for col in categorical_cols):
		raise ValueError("Some categorical columns are missing from DataFrame.")

	df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

	df_encoded.columns = (
		df_encoded.columns.str.replace("[^a-zA-Z0-9]", "_", regex=True)
		.str.replace("__", "_", regex=True)
		.str.strip()
	)

	y = df_encoded[target_col]
	X = df_encoded.drop(columns=[target_col])

	return X, y


def enhance_application_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Enhance DataFrame with advanced credit risk features, building on the
	already cleaned data from clean_creditrisk function.

	Args:
		df: DataFrame already processed by clean_creditrisk

	Returns:
		DataFrame with additional engineered features
	"""
	df = df.copy()

	for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
	            'AMT_GOODS_PRICE']:
		if col in df.columns:
			for other_col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
			                  'AMT_GOODS_PRICE']:
				if other_col in df.columns and col != other_col:
					ratio_name = f"{col}_TO_{other_col}_RATIO"
					df[ratio_name] = df[col] / df[other_col].replace(0, np.nan)

	df["DEBT_BURDEN_RATIO"] = df["AMT_ANNUITY"] * 12 / df["AMT_INCOME_TOTAL"]
	df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

	df["LONG_EMPLOYMENT"] = (df["YEARS_EMPLOYED"] >= 5).astype(int)

	income_risk_map = {
		"Working"      : 1, "Commercial associate": 2, "Pensioner": 3,
		"State servant": 4, "Unemployed": 5, "Student": 5,
		"Businessman"  : 2, "Maternity leave": 5
	}
	df["INCOME_TYPE_RISK"] = df["NAME_INCOME_TYPE"].map(income_risk_map)

	df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"]
	df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
	df["CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / df["CNT_FAM_MEMBERS"]

	doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
	if doc_cols:
		df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
		df["DOCUMENT_SUBMISSION_RATIO"] = df["DOCUMENT_COUNT"] / len(doc_cols)

	ext_source_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

	if len(ext_source_cols) > 0:
		df["EXT_SOURCE_MEAN"] = df[ext_source_cols].mean(axis=1)

		if len(ext_source_cols) > 1:
			df["EXT_SOURCE_STD"] = df[ext_source_cols].std(axis=1)

		if len(ext_source_cols) == 3:
			df["EXT_SOURCE_PROD"] = (
					df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
			)

		for i, col1 in enumerate(ext_source_cols):
			for col2 in ext_source_cols[i + 1:]:
				df[f"{col1}_{col2}_INTERACTION"] = df[col1] * df[col2]

	df["AGE_RANGE"] = pd.cut(
		df["AGE_YEARS"],
		bins=[0, 25, 35, 45, 55, 65, 100],
		labels=[1, 2, 3, 4, 5, 6]
	)

	for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
	            "AMT_GOODS_PRICE"]:
		if col in df.columns:
			df[f"{col}_LOG"] = np.log1p(df[col].replace(0, 0.001))

	for col in ext_source_cols:
		df[f"{col}_SQ"] = df[col] ** 2
		df[f"{col}_SQRT"] = np.sqrt(df[col].clip(lower=0))

	df["CREDIT_INCOME_PERCENT_BINS"] = pd.qcut(
		df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan),
		q=10, labels=False, duplicates='drop'
	).fillna(-1).astype(int)

	for col in df.columns:
		if df[col].dtype in [np.float64, np.float32]:
			df[col] = df[col].replace([np.inf, -np.inf], np.nan)

			if col in ext_source_cols or 'RATIO' in col:
				df[f"{col}_MISSING"] = df[col].isna().astype(int)

			df[col] = df[col].fillna(df[col].median())

	return df


def enhance_bureau_features(
    bureau_df: pd.DataFrame, active_status: str = None
) -> pd.DataFrame:
    """
    Calculate aggregated features from bureau data for credit risk modeling.

    Args:
        bureau_df: Bureau DataFrame with credit history from other institutions
        active_status: Filter by credit status (Active, Closed, etc.) or None

    Returns:
        DataFrame with aggregated bureau features by SK_ID_CURR
    """
    if active_status:
        filtered_df = bureau_df[
            bureau_df["CREDIT_ACTIVE"] == active_status
        ].copy()
    else:
        filtered_df = bureau_df.copy()

    agg_funcs = {
        "DAYS_CREDIT": ["min", "max", "count"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
        "AMT_CREDIT_MAX_OVERDUE": ["max"],
        "CREDIT_DAY_OVERDUE": ["max", "sum"],
        "DAYS_CREDIT_ENDDATE": ["min", "max"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "AMT_ANNUITY": ["sum"],
    }

    credit_type_dummies = pd.get_dummies(filtered_df["CREDIT_TYPE"])
    for credit_type in credit_type_dummies.columns:
        filtered_df[f"CREDIT_TYPE_{credit_type}"] = (
	        credit_type_dummies)[credit_type]
        agg_funcs[f"CREDIT_TYPE_{credit_type}"] = ["sum"]

    df_agg = filtered_df.groupby("SK_ID_CURR").agg(agg_funcs).reset_index()

    df_agg.columns = [
        "SK_ID_CURR" if col[0] == "SK_ID_CURR" else f"{col[0]}_{col[1]}"
        for col in df_agg.columns.values
    ]

    df_agg["CREDIT_ACTIVE_COUNT"] = df_agg["DAYS_CREDIT_count"]

    df_agg["DEBT_TO_CREDIT_RATIO"] = (
        df_agg["AMT_CREDIT_SUM_DEBT_sum"] /
        df_agg["AMT_CREDIT_SUM_sum"].replace(0, np.nan)
    )

    df_agg["OVERDUE_RATIO"] = (
        df_agg["AMT_CREDIT_SUM_OVERDUE_sum"] /
        df_agg["AMT_CREDIT_SUM_sum"].replace(0, np.nan)
    )

    df_agg["AVERAGE_LOAN_AMOUNT"] = (
        df_agg["AMT_CREDIT_SUM_sum"] /
        df_agg["CREDIT_ACTIVE_COUNT"].replace(0, np.nan)
    )

    df_agg["PROLONGED_CREDIT_RATIO"] = (
        df_agg["CNT_CREDIT_PROLONG_sum"] /
        df_agg["CREDIT_ACTIVE_COUNT"].replace(0, np.nan)
    )

    df_agg["ANNUITY_TO_CREDIT_RATIO"] = (
        df_agg["AMT_ANNUITY_sum"] /
        df_agg["AMT_CREDIT_SUM_sum"].replace(0, np.nan)
    )

    df_agg["CREDIT_HISTORY_LENGTH"] = abs(df_agg["DAYS_CREDIT_min"]) / 365.25

    df_agg["RECENT_CREDIT_ACTIVITY"] = abs(df_agg["DAYS_CREDIT_max"]) / 30

    df_agg["HAS_OVERDUE"] = (
		    df_agg["AMT_CREDIT_SUM_OVERDUE_sum"] > 0).astype(int)

    df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
    df_agg = df_agg.fillna(0)

    return df_agg



def enhance_cc_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate credit card balance features for credit risk modeling.

    Args:
        df: DataFrame containing credit card balance data

    Returns:
        DataFrame with aggregated credit card features by SK_ID_CURR
    """
    df_agg = df.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": ["nunique"],
        "MONTHS_BALANCE": ["min", "max"],
        "AMT_BALANCE": ["mean", "max", "sum", "std"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
        "AMT_DRAWINGS_ATM_CURRENT": ["sum", "mean"],
        "AMT_DRAWINGS_CURRENT": ["sum", "mean"],
        "AMT_DRAWINGS_OTHER_CURRENT": ["sum"],
        "AMT_DRAWINGS_POS_CURRENT": ["sum"],
        "AMT_PAYMENT_CURRENT": ["sum", "mean"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["sum"],
        "AMT_TOTAL_RECEIVABLE": ["sum"],
        "CNT_DRAWINGS_CURRENT": ["sum"],
        "CNT_DRAWINGS_ATM_CURRENT": ["sum"],
        "SK_DPD": ["max", "sum"],
        "SK_DPD_DEF": ["max", "sum"],
    })

    df_agg.columns = ["_".join(col).strip().upper() for col in df_agg.columns]
    df_agg.rename(columns={"SK_ID_CURR_": "SK_ID_CURR"}, inplace=True)

    df_agg["CREDIT_UTILIZATION_RATIO"] = (
        df_agg["AMT_BALANCE_MEAN"] /
        df_agg["AMT_CREDIT_LIMIT_ACTUAL_MEAN"].replace(0, np.nan)
    )

    df_agg["TOTAL_DRAWINGS_RATIO"] = (
        df_agg["AMT_DRAWINGS_ATM_CURRENT_SUM"] +
        df_agg["AMT_DRAWINGS_CURRENT_SUM"] +
        df_agg["AMT_DRAWINGS_OTHER_CURRENT_SUM"] +
        df_agg["AMT_DRAWINGS_POS_CURRENT_SUM"]
    ) / df_agg["AMT_CREDIT_LIMIT_ACTUAL_MEAN"].replace(0, np.nan)

    df_agg["OVERDUE_RATIO"] = (
        df_agg["SK_DPD_DEF_MAX"] /
        (df_agg["SK_DPD_DEF_MAX"] + df_agg["SK_DPD_MAX"]).replace(0, np.nan)
    )

    df_agg["ATM_DRAWING_RATIO"] = (
        df_agg["AMT_DRAWINGS_ATM_CURRENT_SUM"] /
        (df_agg["AMT_DRAWINGS_CURRENT_SUM"] + 0.01)
    )

    df_agg["BALANCE_VOLATILITY"] = (
        df_agg["AMT_BALANCE_STD"] /
        (df_agg["AMT_BALANCE_MEAN"] + 0.01)
    )

    df_agg["CC_COUNT"] = df_agg["SK_ID_PREV_NUNIQUE"]

    df_agg["HAS_DPD"] = (df_agg["SK_DPD_MAX"] > 0).astype(int)

    df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
    df_agg = df_agg.fillna(0)

    return df_agg


def enhance_installments_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Calculate aggregated features from installments payment data."""

	df["DAYS_LATE"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
	df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, 1)
	df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]

	df_agg = df.groupby("SK_ID_CURR").agg({
		"SK_ID_PREV"            : ["nunique"],
		"NUM_INSTALMENT_VERSION": ["max"],
		"NUM_INSTALMENT_NUMBER" : ["sum"],
		"DAYS_INSTALMENT"       : ["min", "max"],
		"DAYS_ENTRY_PAYMENT"    : ["min", "max"],
		"AMT_INSTALMENT"        : ["min", "max", "mean", "sum"],
		"AMT_PAYMENT"           : ["min", "max", "mean", "sum"],
		"DAYS_LATE"             : ["min", "max", "mean", "sum",
		                           lambda x: (x > 0).sum()],
		"PAYMENT_RATIO"         : ["min", "max", "mean"],
		"PAYMENT_DIFF"          : ["sum"]
	})

	df_agg.columns = ["_".join(col).upper() for col in df_agg.columns]
	df_agg.reset_index(inplace=True)

	df_agg.rename(columns={
		"DAYS_LATE_<LAMBDA_0>": "LATE_PAYMENT_COUNT"
	}, inplace=True)

	df_agg["LATE_PAYMENT_RATIO"] = (
			df_agg["LATE_PAYMENT_COUNT"] /
			df_agg["NUM_INSTALMENT_NUMBER_SUM"]
	)

	df_agg["AVG_PAYMENT_DELAY"] = (
			df_agg["DAYS_LATE_SUM"] /
			df_agg["NUM_INSTALMENT_NUMBER_SUM"]
	)

	df_agg["TOTAL_PAYMENT_RATIO"] = (
			df_agg["AMT_PAYMENT_SUM"] /
			df_agg["AMT_INSTALMENT_SUM"]
	)

	df_agg["RECENT_INSTALMENT_TREND"] = (
			df_agg["DAYS_INSTALMENT_MAX"] -
			df_agg["DAYS_INSTALMENT_MIN"]
	)

	df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
	df_agg = df_agg.fillna(0)

	return df_agg


def enhance_poscash_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Calculate key POS cash balance features for credit risk prediction."""

	df["HAS_DPD"] = (df["SK_DPD"] > 0).astype(int)
	df["COMPLETED_LOAN"] = (
			df["CNT_INSTALMENT_FUTURE"] == df["CNT_INSTALMENT"]).astype(int)

	pos_cash_agg = df.groupby("SK_ID_CURR").agg({
		"SK_ID_PREV"           : ["nunique"],
		"SK_DPD"               : ["max", "sum"],
		"SK_DPD_DEF"           : ["max"],
		"CNT_INSTALMENT_FUTURE": ["sum"],
		"CNT_INSTALMENT"       : ["sum"],
		"HAS_DPD"              : ["sum"],
		"COMPLETED_LOAN"       : ["sum"],
		"NAME_CONTRACT_STATUS" : lambda x: x.value_counts().get("Completed", 0)
	})

	pos_cash_agg.columns = [
		"POS_" + "_".join(col).upper() if isinstance(col, tuple)
		else f"POS_{col}" for col in pos_cash_agg.columns
	]

	pos_cash_agg.rename(columns={
		"POS_NAME_CONTRACT_STATUS_<LAMBDA>": "POS_NAME_CONTRACT_COMPLETED_COUNT"
	}, inplace=True)

	pos_cash_agg["POS_DPD_RATE"] = (
			pos_cash_agg["POS_HAS_DPD_SUM"] /
			pos_cash_agg["POS_SK_ID_PREV_NUNIQUE"]
	).replace(np.inf, np.nan)

	pos_cash_agg = pos_cash_agg.replace([np.inf, -np.inf], np.nan)
	pos_cash_agg = pos_cash_agg.fillna(0)

	return pos_cash_agg


def enhance_previous_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Calculate key features from previous applications for credit risk modeling.

	Args:
		df: DataFrame containing previous applications data

	Returns:
		DataFrame with most predictive previous application features
	"""

	df["APP_CREDIT_PERC"] = df["AMT_APPLICATION"] / df["AMT_CREDIT"]
	df["AMT_CREDIT_GOODS_PERC"] = (
			df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
	)
	df["TERM_YEARS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)

	df["IS_APPROVED"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
	df["IS_REFUSED"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

	agg_funcs = {
		"SK_ID_PREV"           : ["nunique"],
		"AMT_CREDIT"           : ["mean", "sum"],
		"AMT_ANNUITY"          : ["mean", "sum"],
		"APP_CREDIT_PERC"      : ["mean"],
		"AMT_CREDIT_GOODS_PERC": ["mean"],
		"TERM_YEARS"           : ["mean"],
		"DAYS_DECISION"        : ["min", "max"],
		"CNT_PAYMENT"          : ["mean"],
		"IS_APPROVED"          : ["mean", "sum"],
		"IS_REFUSED"           : ["mean", "sum"],
		"NAME_CONTRACT_TYPE"   : lambda x: x.value_counts().get(
			"Consumer loans", 0),
		"NAME_YIELD_GROUP"     : lambda x: x.value_counts().get("high", 0),
	}

	df_agg = df.groupby("SK_ID_CURR").agg(agg_funcs)

	df_agg.columns = [
		f"PREV_{col[0]}_{col[1].upper()}" if isinstance(col[1], str)
		else f"PREV_{col[0]}" for col in df_agg.columns
	]

	df_agg.rename(columns={
		"PREV_NAME_CONTRACT_TYPE_<LAMBDA>": "PREV_CONSUMER_LOANS_COUNT",
		"PREV_NAME_YIELD_GROUP_<LAMBDA>"  : "PREV_HIGH_YIELD_COUNT"
	}, inplace=True)

	df_agg["PREV_APPROVAL_RATE"] = (
			df_agg["PREV_IS_APPROVED_SUM"] / df_agg["PREV_SK_ID_PREV_NUNIQUE"]
	)
	df_agg["PREV_REFUSAL_RATE"] = (
			df_agg["PREV_IS_REFUSED_SUM"] / df_agg["PREV_SK_ID_PREV_NUNIQUE"]
	)
	df_agg["PREV_AVG_APPLICATION_PERIOD"] = (
			(df_agg["PREV_DAYS_DECISION_MAX"] - df_agg["PREV_DAYS_DECISION_MIN"]
			).abs() / 30)

	df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
	df_agg = df_agg.fillna(0)

	return df_agg.reset_index()


def detect_outliers(
	df: pd.DataFrame,
	threshold: float = 3.0,
	exclude_cols: list = None,
	mark_outliers: bool = False,
	method: str = "zscore",
	column_name: str = "IS_OUTLIER"
) -> tuple[pd.DataFrame, dict]:
	"""
	Detect extreme outliers in DataFrame features and optionally mark them.

	Args:
		df: Input DataFrame to check for outliers
		threshold: Threshold for outlier detection (default: 3.0 for z-score,
				  1.5 for IQR method)
		exclude_cols: List of column names to exclude from outlier detection
		mark_outliers: Whether to add IS_OUTLIER column to mark records
		with outliers
		method: Detection method - "zscore" or "iqr"

	Returns:
		tuple: (DataFrame with optional outlier marking,
				Dictionary with outlier counts by feature)
	"""

	result_df = df.copy()

	if exclude_cols is None:
		exclude_cols = []

	for col in df.columns:
		if col in exclude_cols:
			continue
		if df[col].dtype == 'object' or df[col].dtype == 'category':
			exclude_cols.append(col)
		if col == 'TARGET' or col.startswith('IS_'):
			exclude_cols.append(col)

	num_cols = [col for col in df.columns if col not in exclude_cols
	            and df[col].dtype in ['int64', 'float64']]

	outlier_counts = {}

	if mark_outliers:
		all_outliers = pd.DataFrame(False,
		                            index=df.index,
		                            columns=num_cols)

	for col in num_cols:
		if df[col].isna().all():
			continue

		valid_data = df[col].dropna()

		if method == "zscore":
			mean_val = valid_data.mean()
			std_val = valid_data.std()

			if std_val == 0:
				continue

			z_scores = np.abs((valid_data - mean_val) / std_val)
			outliers = valid_data[z_scores > threshold]

		elif method == "iqr":
			q1 = valid_data.quantile(0.25)
			q3 = valid_data.quantile(0.75)
			iqr = q3 - q1

			lower_bound = q1 - threshold * iqr
			upper_bound = q3 + threshold * iqr

			outliers = valid_data[(valid_data < lower_bound) |
			                      (valid_data > upper_bound)]
		else:
			raise ValueError("Method must be 'zscore' or 'iqr'")

		if len(outliers) > 0:
			outlier_counts[col] = len(outliers)

			if mark_outliers:
				if method == "zscore":
					all_outliers[col] = np.abs(
						(df[col] - mean_val) / std_val
					) > threshold
				else:
					all_outliers[col] = (
							(df[col] < lower_bound) | (df[col] > upper_bound)
					)

	if mark_outliers and not all_outliers.empty:
		result_df[column_name] = all_outliers.any(axis=1).astype(int)

	outlier_counts = dict(sorted(outlier_counts.items(),
	                             key=lambda x: x[1], reverse=True))

	print(f"Found {len(outlier_counts)} features with outliers")
	print(f"Total outlier records: {sum(outlier_counts.values())}")

	if outlier_counts:
		print("\nTop features with outliers:")
		for i, (col, count) in enumerate(list(outlier_counts.items())[:10]):
			print(f"{i + 1}. {col}: {count} outliers")

	return result_df, outlier_counts


def remove_highly_correlated_features(
	df: pd.DataFrame, cat_features: list, threshold: float = 0.9
) -> list:
	"""
	Identify numerical features that are highly correlated.

	Args:
		df: DataFrame containing features
		cat_features: List of categorical feature names to exclude
		threshold: Correlation threshold for feature removal (default: 0.9)

	Returns:
		List of feature names to drop due to high correlation
	"""
	num_df = df.drop(columns=cat_features)

	corr_matrix = num_df.corr().abs()

	upper_triangle = corr_matrix.where(
		np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_)
	)

	to_drop = [
		column
		for column in upper_triangle.columns
		if any(upper_triangle[column] > threshold)
	]

	print(
		f"Number of highly correlated features to be removed: {len(to_drop)}"
	)
	return to_drop
