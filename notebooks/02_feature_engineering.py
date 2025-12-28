{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Feature Engineering\n",
    "## Creating and testing new features\n",
    "\n",
    "**Author:** Your Name  \n",
    "**Date:** 2024-12-28  \n",
    "**Objective:** Engineer new features and test their impact on model performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, PolynomialFeatures\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report, roc_auc_score\n",
    "\n",
    "%matplotlib inline\n",
    "print(\"✅ Setup complete\")"
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
    "# Load preprocessed data\n",
    "# df = pd.read_csv('../data/02-preprocessed/data.csv')\n",
    "\n",
    "# Demo data\n",
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
    "## 2. Baseline Model Performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare baseline features\n",
    "X_baseline = df.drop('target', axis=1)\n",
    "X_baseline = pd.get_dummies(X_baseline, drop_first=True)  # Encode categoricals\n",
    "y = df['target']\n",
    "\n",
    "# Train-test split\n",
    "X_train_base, X_test_base, y_train, y_test = train_test_split(\n",
    "    X_baseline, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "# Scale features\n",
    "scaler = StandardScaler()\n",
    "X_train_base_scaled = scaler.fit_transform(X_train_base)\n",
    "X_test_base_scaled = scaler.transform(X_test_base)\n",
    "\n",
    "# Train baseline model\n",
    "baseline_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
    "baseline_model.fit(X_train_base_scaled, y_train)\n",
    "\n",
    "# Evaluate\n",
    "y_pred_base = baseline_model.predict(X_test_base_scaled)\n",
    "y_pred_proba_base = baseline_model.predict_proba(X_test_base_scaled)[:, 1]\n",
    "\n",
    "baseline_auc = roc_auc_score(y_test, y_pred_proba_base)\n",
    "\n",
    "print(\"Baseline Model Performance:\")\n",
    "print(\"=\"*50)\n",
    "print(f\"ROC-AUC: {baseline_auc:.4f}\")\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(y_test, y_pred_base))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create new features\n",
    "df_engineered = df.copy()\n",
    "\n",
    "# 1. Interaction features\n",
    "df_engineered['feat1_x_feat2'] = df['feature_1'] * df['feature_2']\n",
    "df_engineered['feat1_div_feat2'] = df['feature_1'] / (df['feature_2'] + 1e-5)\n",
    "\n",
    "# 2. Polynomial features\n",
    "df_engineered['feat1_squared'] = df['feature_1'] ** 2\n",
    "df_engineered['feat2_squared'] = df['feature_2'] ** 2\n",
    "df_engineered['feat1_cubed'] = df['feature_1'] ** 3\n",
    "\n",
    "# 3. Log transformations\n",
    "df_engineered['feat1_log'] = np.log1p(np.abs(df['feature_1']))\n",
    "df_engineered['feat2_log'] = np.log1p(np.abs(df['feature_2']))\n",
    "\n",
    "# 4. Binning\n",
    "df_engineered['feat1_binned'] = pd.cut(df['feature_1'], bins=5, labels=False)\n",
    "\n",
    "# 5. Aggregation features (if you have grouping variables)\n",
    "# df_engineered['feat1_by_group_mean'] = df.groupby('feature_3')['feature_1'].transform('mean')\n",
    "\n",
    "print(f\"Original features: {df.shape[1]}\")\n",
    "print(f\"Engineered features: {df_engineered.shape[1]}\")\n",
    "print(f\"New features added: {df_engineered.shape[1] - df.shape[1]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature Importance Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare engineered features\n",
    "X_engineered = df_engineered.drop('target', axis=1)\n",
    "X_engineered = pd.get_dummies(X_engineered, drop_first=True)\n",
    "\n",
    "# Train-test split\n",
    "X_train_eng, X_test_eng, y_train, y_test = train_test_split(\n",
    "    X_engineered, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "# Scale\n",
    "scaler_eng = StandardScaler()\n",
    "X_train_eng_scaled = scaler_eng.fit_transform(X_train_eng)\n",
    "X_test_eng_scaled = scaler_eng.transform(X_test_eng)\n",
    "\n",
    "# Train model with engineered features\n",
    "model_eng = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
    "model_eng.fit(X_train_eng_scaled, y_train)\n",
    "\n",
    "# Feature importance\n",
    "feature_importance = pd.DataFrame({\n",
    "    'feature': X_engineered.columns,\n",
    "    'importance': model_eng.feature_importances_\n",
    "}).sort_values('importance', ascending=False)\n",
    "\n",
    "# Plot top 15 features\n",
    "plt.figure(figsize=(10, 8))\n",
    "top_features = feature_importance.head(15)\n",
    "plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')\n",
    "plt.yticks(range(len(top_features)), top_features['feature'])\n",
    "plt.xlabel('Importance', fontsize=12)\n",
    "plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')\n",
    "plt.gca().invert_yaxis()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\nTop 10 Most Important Features:\")\n",
    "print(feature_importance.head(10))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Model Performance Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate engineered model\n",
    "y_pred_eng = model_eng.predict(X_test_eng_scaled)\n",
    "y_pred_proba_eng = model_eng.predict_proba(X_test_eng_scaled)[:, 1]\n",
    "\n",
    "engineered_auc = roc_auc_score(y_test, y_pred_proba_eng)\n",
    "\n",
    "print(\"Model with Engineered Features:\")\n",
    "print(\"=\"*50)\n",
    "print(f\"ROC-AUC: {engineered_auc:.4f}\")\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(y_test, y_pred_eng))\n",
    "\n",
    "# Comparison\n",
    "print(\"\\n\" + \"=\"*50)\n",
    "print(\"PERFORMANCE COMPARISON\")\n",
    "print(\"=\"*50)\n",
    "print(f\"Baseline ROC-AUC:    {baseline_auc:.4f}\")\n",
    "print(f\"Engineered ROC-AUC:  {engineered_auc:.4f}\")\n",
    "improvement = ((engineered_auc - baseline_auc) / baseline_auc) * 100\n",
    "print(f\"Improvement:         {improvement:+.2f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Feature Selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Select top features based on importance\n",
    "top_n = 10\n",
    "selected_features = feature_importance.head(top_n)['feature'].tolist()\n",
    "\n",
    "print(f\"Selected Top {top_n} Features:\")\n",
    "for i, feat in enumerate(selected_features, 1):\n",
    "    print(f\"{i}. {feat}\")\n",
    "\n",
    "# Train model with selected features only\n",
    "X_train_selected = X_train_eng[selected_features]\n",
    "X_test_selected = X_test_eng[selected_features]\n",
    "\n",
    "scaler_selected = StandardScaler()\n",
    "X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)\n",
    "X_test_selected_scaled = scaler_selected.transform(X_test_selected)\n",
    "\n",
    "model_selected = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
    "model_selected.fit(X_train_selected_scaled, y_train)\n",
    "\n",
    "y_pred_selected = model_selected.predict(X_test_selected_scaled)\n",
    "y_pred_proba_selected = model_selected.predict_proba(X_test_selected_scaled)[:, 1]\n",
    "\n",
    "selected_auc = roc_auc_score(y_test, y_pred_proba_selected)\n",
    "\n",
    "print(f\"\\nModel with Top {top_n} Features:\")\n",
    "print(f\"ROC-AUC: {selected_auc:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Save Engineered Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save engineered dataset\n",
    "output_path = '../data/03-features/'\n",
    "import os\n",
    "os.makedirs(output_path, exist_ok=True)\n",
    "\n",
    "df_engineered.to_csv(f'{output_path}engineered_features.csv', index=False)\n",
    "print(f\"✅ Engineered features saved to {output_path}\")\n",
    "\n",
    "# Save selected feature list\n",
    "with open(f'{output_path}selected_features.txt', 'w') as f:\n",
    "    for feat in selected_features:\n",
    "        f.write(f\"{feat}\\n\")\n",
    "print(f\"✅ Selected features list saved\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Summary & Recommendations\n",
    "\n",
    "### Feature Engineering Results:\n",
    "- Created X new features\n",
    "- Model performance improved by Y%\n",
    "- Top features: [list]\n",
    "\n",
    "### Next Steps:\n",
    "1. Move feature engineering logic to `src/pipelines/feature_eng_pipeline.py`\n",
    "2. Test on validation set\n",
    "3. Hyperparameter tuning with engineered features\n",
    "4. Cross-validation to ensure robustness"
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