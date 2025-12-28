{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Experimentation\n",
    "## Testing different algorithms and hyperparameters\n",
    "\n",
    "**Author:** Your Name  \n",
    "**Date:** 2024-12-28  \n",
    "**Objective:** Compare multiple ML algorithms and select the best performing model"
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
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    roc_auc_score, roc_curve, confusion_matrix, classification_report\n",
    ")\n",
    "\n",
    "# Models\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.svm import SVC\n",
    "from xgboost import XGBClassifier\n",
    "from lightgbm import LGBMClassifier\n",
    "\n",
    "%matplotlib inline\n",
    "print(\"✅ Setup complete\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Engineered Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\n",
    "# df = pd.read_csv('../data/03-features/engineered_features.csv')\n",
    "\n",
    "# Demo data\n",
    "np.random.seed(42)\n",
    "df = pd.DataFrame({\n",
    "    'feature_1': np.random.randn(1000),\n",
    "    'feature_2': np.random.randn(1000) * 2 + 1,\n",
    "    'feature_3': np.random.randn(1000) * 0.5,\n",
    "    'target': np.random.randint(0, 2, 1000)\n",
    "})\n",
    "\n",
    "X = df.drop('target', axis=1)\n",
    "y = df['target']\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "# Scale features\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(f\"Training set: {X_train.shape}\")\n",
    "print(f\"Test set: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Model Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define models to test\n",
    "models = {\n",
    "    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),\n",
    "    'Decision Tree': DecisionTreeClassifier(random_state=42),\n",
    "    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),\n",
    "    'Gradient Boosting': GradientBoostingClassifier(random_state=42),\n",
    "    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),\n",
    "    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),\n",
    "    'SVM': SVC(random_state=42, probability=True)\n",
    "}\n",
    "\n",
    "# Train and evaluate each model\n",
    "results = []\n",
    "\n",
    "for name, model in models.items():\n",
    "    print(f\"Training {name}...\")\n",
    "    \n",
    "    # Train\n",
    "    model.fit(X_train_scaled, y_train)\n",
    "    \n",
    "    # Predict\n",
    "    y_pred = model.predict(X_test_scaled)\n",
    "    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None\n",
    "    \n",
    "    # Calculate metrics\n",
    "    accuracy = accuracy_score(y_test, y_pred)\n",
    "    precision = precision_score(y_test, y_pred, average='weighted')\n",
    "    recall = recall_score(y_test, y_pred, average='weighted')\n",
    "    f1 = f1_score(y_test, y_pred, average='weighted')\n",
    "    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None\n",
    "    \n",
    "    # Cross-validation score\n",
    "    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')\n",
    "    cv_mean = cv_scores.mean()\n",
    "    cv_std = cv_scores.std()\n",
    "    \n",
    "    results.append({\n",
    "        'Model': name,\n",
    "        'Accuracy': accuracy,\n",
    "        'Precision': precision,\n",
    "        'Recall': recall,\n",
    "        'F1-Score': f1,\n",
    "        'ROC-AUC': auc,\n",
    "        'CV Mean': cv_mean,\n",
    "        'CV Std': cv_std\n",
    "    })\n",
    "\n",
    "# Create results DataFrame\n",
    "results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)\n",
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"MODEL COMPARISON RESULTS\")\n",
    "print(\"=\"*80)\n",
    "display(results_df.style.background_gradient(cmap='YlGn', subset=['Accuracy', 'ROC-AUC']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Visualize Model Performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Bar plot comparison\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
    "\n",
    "metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']\n",
    "for idx, metric in enumerate(metrics):\n",
    "    ax = axes[idx // 2, idx % 2]\n",
    "    data = results_df.sort_values(metric, ascending=True)\n",
    "    ax.barh(data['Model'], data[metric], color='steelblue')\n",
    "    ax.set_xlabel(metric, fontsize=12)\n",
    "    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')\n",
    "    ax.set_xlim([0, 1])\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. ROC Curves"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot ROC curves for all models\n",
    "plt.figure(figsize=(10, 8))\n",
    "\n",
    "for name, model in models.items():\n",
    "    if hasattr(model, 'predict_proba'):\n",
    "        model.fit(X_train_scaled, y_train)\n",
    "        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]\n",
    "        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)\n",
    "        auc = roc_auc_score(y_test, y_pred_proba)\n",
    "        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)\n",
    "\n",
    "plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=2)\n",
    "plt.xlabel('False Positive Rate', fontsize=12)\n",
    "plt.ylabel('True Positive Rate', fontsize=12)\n",
    "plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')\n",
    "plt.legend(loc='lower right')\n",
    "plt.grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Hyperparameter Tuning - Best Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Select best model (highest ROC-AUC)\n",
    "best_model_name = results_df.iloc[0]['Model']\n",
    "print(f\"Best Model: {best_model_name}\")\n",
    "print(f\"ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}\")\n",
    "\n",
    "# Hyperparameter tuning for Random Forest (example)\n",
    "if 'Random Forest' in best_model_name:\n",
    "    param_grid = {\n",
    "        'n_estimators': [100, 200, 300],\n",
    "        'max_depth': [10, 20, 30, None],\n",
    "        'min_samples_split': [2, 5, 10],\n",
    "        'min_samples_leaf': [1, 2, 4]\n",
    "    }\n",
    "    \n",
    "    print(\"\\nPerforming GridSearchCV...\")\n",
    "    rf = RandomForestClassifier(random_state=42)\n",
    "    grid_search = GridSearchCV(\n",
    "        rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1\n",
    "    )\n",
    "    grid_search.fit(X_train_scaled, y_train)\n",
    "    \n",
    "    print(\"\\nBest Parameters:\")\n",
    "    print(grid_search.best_params_)\n",
    "    print(f\"\\nBest CV ROC-AUC: {grid_search.best_score_:.4f}\")\n",
    "    \n",
    "    # Evaluate tuned model\n",
    "    best_model = grid_search.best_estimator_\n",
    "    y_pred_tuned = best_model.predict(X_test_scaled)\n",
    "    y_pred_proba_tuned = best_model.predict_proba(X_test_scaled)[:, 1]\n",
    "    \n",
    "    tuned_auc = roc_auc_score(y_test, y_pred_proba_tuned)\n",
    "    print(f\"Test ROC-AUC after tuning: {tuned_auc:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Final Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Detailed evaluation of best model\n",
    "print(\"Final Model Classification Report:\")\n",
    "print(\"=\"*50)\n",
    "print(classification_report(y_test, y_pred_tuned))\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(y_test, y_pred_tuned)\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)\n",
    "plt.xlabel('Predicted', fontsize=12)\n",
    "plt.ylabel('Actual', fontsize=12)\n",
    "plt.title('Confusion Matrix - Best Model', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Save Best Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "import os\n",
    "\n",
    "# Save model\n",
    "model_dir = '../models/'\n",
    "os.makedirs(model_dir, exist_ok=True)\n",
    "\n",
    "joblib.dump(best_model, f'{model_dir}best_model.pkl')\n",
    "joblib.dump(scaler, f'{model_dir}scaler.pkl')\n",
    "\n",
    "print(f\"✅ Best model saved to {model_dir}\")\n",
    "print(f\"Model: {best_model_name}\")\n",
    "print(f\"Test ROC-AUC: {tuned_auc:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Summary & Next Steps\n",
    "\n",
    "### Results Summary:\n",
    "- **Best Model:** [Model Name]\n",
    "- **Test ROC-AUC:** [Score]\n",
    "- **Key Hyperparameters:** [Parameters]\n",
    "\n",
    "### Next Steps:\n",
    "1. Move best model configuration to `src/pipelines/training_pipeline.py`\n",
    "2. Set up MLflow experiment tracking\n",
    "3. Implement model monitoring\n",
    "4. Deploy to production\n",
    "5. Set up A/B testing framework"
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