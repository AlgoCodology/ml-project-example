{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis\n",
    "## Initial data exploration and understanding\n",
    "\n",
    "**Author:** Your Name  \n",
    "**Date:** 2024-12-28  \n",
    "**Objective:** Explore raw data, understand distributions, identify patterns and anomalies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup\n",
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "\n",
    "# Plotting configuration\n",
    "plt.style.use('seaborn-v0_8-darkgrid')\n",
    "sns.set_palette('husl')\n",
    "%matplotlib inline\n",
    "\n",
    "# Display settings\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', 100)\n",
    "\n",
    "print(\"✅ Imports successful\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load raw data\n",
    "data_path = Path('../data/01-raw')\n",
    "\n",
    "# Option 1: Local CSV\n",
    "# df = pd.read_csv(data_path / 'data.csv')\n",
    "\n",
    "# Option 2: From Databricks\n",
    "# from src.utils.databricks_loader import load_from_databricks\n",
    "# from src.utils.config import load_config\n",
    "# config = load_config('../config/databricks.yaml')\n",
    "# df = load_from_databricks(config, catalog='prod_ml', schema='ml_features', table='raw_data')\n",
    "\n",
    "# For demo, create sample data\n",
    "np.random.seed(42)\n",
    "df = pd.DataFrame({\n",
    "    'feature_1': np.random.randn(1000),\n",
    "    'feature_2': np.random.randn(1000) * 2 + 1,\n",
    "    'feature_3': np.random.choice(['A', 'B', 'C'], 1000),\n",
    "    'target': np.random.randint(0, 2, 1000)\n",
    "})\n",
    "\n",
    "print(f\"Data shape: {df.shape}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Basic Information"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dataset info\n",
    "print(\"Dataset Info:\")\n",
    "print(\"=\"*50)\n",
    "df.info()\n",
    "\n",
    "print(\"\\nSummary Statistics:\")\n",
    "print(\"=\"*50)\n",
    "display(df.describe())\n",
    "\n",
    "print(\"\\nMissing Values:\")\n",
    "print(\"=\"*50)\n",
    "missing = df.isnull().sum()\n",
    "missing_pct = (missing / len(df)) * 100\n",
    "missing_df = pd.DataFrame({\n",
    "    'Missing Count': missing,\n",
    "    'Percentage': missing_pct\n",
    "})\n",
    "display(missing_df[missing_df['Missing Count'] > 0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Target Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Target distribution\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Count plot\n",
    "df['target'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')\n",
    "axes[0].set_title('Target Distribution (Counts)', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('Target Class')\n",
    "axes[0].set_ylabel('Count')\n",
    "\n",
    "# Pie chart\n",
    "df['target'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90)\n",
    "axes[1].set_title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')\n",
    "axes[1].set_ylabel('')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\nTarget Value Counts:\")\n",
    "print(df['target'].value_counts())\n",
    "print(f\"\\nClass Balance Ratio: {df['target'].value_counts().min() / df['target'].value_counts().max():.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Numerical Features Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Distribution plots for numerical features\n",
    "numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
    "numeric_cols = [col for col in numeric_cols if col != 'target']\n",
    "\n",
    "fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(14, 4*len(numeric_cols)))\n",
    "\n",
    "if len(numeric_cols) == 1:\n",
    "    axes = axes.reshape(1, -1)\n",
    "\n",
    "for idx, col in enumerate(numeric_cols):\n",
    "    # Histogram\n",
    "    axes[idx, 0].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)\n",
    "    axes[idx, 0].set_title(f'{col} - Histogram', fontweight='bold')\n",
    "    axes[idx, 0].set_xlabel(col)\n",
    "    axes[idx, 0].set_ylabel('Frequency')\n",
    "    \n",
    "    # Box plot\n",
    "    axes[idx, 1].boxplot(df[col].dropna(), vert=True)\n",
    "    axes[idx, 1].set_title(f'{col} - Box Plot', fontweight='bold')\n",
    "    axes[idx, 1].set_ylabel(col)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Correlation Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation matrix\n",
    "correlation_matrix = df[numeric_cols + ['target']].corr()\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', \n",
    "            square=True, linewidths=0.5, cbar_kws={\"shrink\": 0.8})\n",
    "plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\nCorrelation with Target:\")\n",
    "print(\"=\"*50)\n",
    "target_corr = correlation_matrix['target'].drop('target').sort_values(ascending=False)\n",
    "display(target_corr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Categorical Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Categorical features analysis\n",
    "categorical_cols = df.select_dtypes(include=['object']).columns\n",
    "\n",
    "if len(categorical_cols) > 0:\n",
    "    fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(12, 4*len(categorical_cols)))\n",
    "    \n",
    "    if len(categorical_cols) == 1:\n",
    "        axes = [axes]\n",
    "    \n",
    "    for idx, col in enumerate(categorical_cols):\n",
    "        df[col].value_counts().plot(kind='bar', ax=axes[idx], color='coral', edgecolor='black')\n",
    "        axes[idx].set_title(f'{col} - Distribution', fontsize=14, fontweight='bold')\n",
    "        axes[idx].set_xlabel(col)\n",
    "        axes[idx].set_ylabel('Count')\n",
    "        axes[idx].tick_params(axis='x', rotation=45)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    # Unique values count\n",
    "    print(\"\\nUnique Values per Categorical Feature:\")\n",
    "    for col in categorical_cols:\n",
    "        print(f\"{col}: {df[col].nunique()} unique values\")\n",
    "else:\n",
    "    print(\"No categorical features found.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Outlier Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Outlier detection using IQR method\n",
    "def detect_outliers_iqr(df, column):\n",
    "    Q1 = df[column].quantile(0.25)\n",
    "    Q3 = df[column].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "    lower_bound = Q1 - 1.5 * IQR\n",
    "    upper_bound = Q3 + 1.5 * IQR\n",
    "    \n",
    "    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]\n",
    "    return outliers, lower_bound, upper_bound\n",
    "\n",
    "print(\"Outlier Analysis:\")\n",
    "print(\"=\"*50)\n",
    "\n",
    "for col in numeric_cols:\n",
    "    outliers, lower, upper = detect_outliers_iqr(df, col)\n",
    "    outlier_pct = (len(outliers) / len(df)) * 100\n",
    "    print(f\"\\n{col}:\")\n",
    "    print(f\"  Outliers: {len(outliers)} ({outlier_pct:.2f}%)\")\n",
    "    print(f\"  Bounds: [{lower:.2f}, {upper:.2f}]\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Feature Relationships with Target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize feature distributions by target class\n",
    "fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 4*len(numeric_cols)))\n",
    "\n",
    "if len(numeric_cols) == 1:\n",
    "    axes = [axes]\n",
    "\n",
    "for idx, col in enumerate(numeric_cols):\n",
    "    for target_value in df['target'].unique():\n",
    "        subset = df[df['target'] == target_value][col].dropna()\n",
    "        axes[idx].hist(subset, bins=30, alpha=0.5, label=f'Target={target_value}')\n",
    "    \n",
    "    axes[idx].set_title(f'{col} Distribution by Target', fontsize=14, fontweight='bold')\n",
    "    axes[idx].set_xlabel(col)\n",
    "    axes[idx].set_ylabel('Frequency')\n",
    "    axes[idx].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Key Findings & Next Steps\n",
    "\n",
    "### Summary of Findings:\n",
    "1. **Data Quality:** [Your observations]\n",
    "2. **Target Distribution:** [Balanced/Imbalanced?]\n",
    "3. **Missing Values:** [Any concerns?]\n",
    "4. **Feature Correlations:** [Strong correlations found?]\n",
    "5. **Outliers:** [Significant outliers?]\n",
    "\n",
    "### Next Steps:\n",
    "- [ ] Handle missing values\n",
    "- [ ] Address class imbalance (if present)\n",
    "- [ ] Feature engineering based on insights\n",
    "- [ ] Outlier treatment strategy\n",
    "- [ ] Feature selection\n",
    "\n",
    "### Recommendations:\n",
    "- [Your recommendations based on analysis]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save processed insights\n",
    "insights = {\n",
    "    'total_rows': len(df),\n",
    "    'total_features': len(df.columns),\n",
    "    'missing_data': df.isnull().sum().to_dict(),\n",
    "    'target_distribution': df['target'].value_counts().to_dict(),\n",
    "    'numeric_features': list(numeric_cols),\n",
    "    'categorical_features': list(categorical_cols)\n",
    "}\n",
    "\n",
    "print(\"✅ EDA Complete!\")\n",
    "print(f\"\\nDataset has {insights['total_rows']} rows and {insights['total_features']} features\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}