"""
Visualization Module
Contains functions for creating visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_correlation_matrix(df, feature_cols):
    """
    Plot correlation matrix heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing features
    feature_cols : list
        List of feature columns to include
    """
    feature_corr = df[feature_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(feature_corr, annot=True, cmap='coolwarm', fmt='.3f', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('images/01_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  [Saved: images/01_correlation_matrix.png]")


def plot_model_comparison(comparison_df):
    """
    Plot comprehensive model comparison
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Dataframe with model comparison metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = comparison_df['Model'].tolist()
    r2_scores = comparison_df['R² Score'].tolist()
    
    # Plot 1: R² Score Comparison
    colors = ['green' if r2 > 0.9 else 'orange' if r2 > 0.7 else 'red' for r2 in r2_scores]
    axes[0, 0].barh(model_names, r2_scores, color=colors)
    axes[0, 0].set_xlabel('R² Score', fontsize=12)
    axes[0, 0].set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Good: 0.7')
    axes[0, 0].axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent: 0.9')
    axes[0, 0].legend()
    
    # Plot 2: RMSE Comparison
    rmse_scores = comparison_df['RMSE (Cr)'].tolist()
    axes[0, 1].barh(model_names, rmse_scores, color='coral')
    axes[0, 1].set_xlabel('RMSE (Rs. Crores)', fontsize=12)
    axes[0, 1].set_title('Model Comparison - RMSE (Lower is Better)', fontsize=14, fontweight='bold')
    
    # Plot 3: MAE Comparison
    mae_scores = comparison_df['MAE (Cr)'].tolist()
    axes[1, 0].barh(model_names, mae_scores, color='skyblue')
    axes[1, 0].set_xlabel('MAE (Rs. Crores)', fontsize=12)
    axes[1, 0].set_title('Model Comparison - MAE (Lower is Better)', fontsize=14, fontweight='bold')
    
    # Plot 4: Cross-Validation R² with error bars
    cv_means = comparison_df['CV R² Mean'].tolist()
    cv_stds = comparison_df['CV R² Std'].tolist()
    axes[1, 1].barh(model_names, cv_means, xerr=[s*2 for s in cv_stds], 
                    color='mediumseagreen', capsize=5)
    axes[1, 1].set_xlabel('Cross-Validation R² (5-fold)', fontsize=12)
    axes[1, 1].set_title('Model Comparison - CV R² with Std Dev', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/02_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  [Saved: images/02_model_comparison.png]")


def plot_train_test_comparison(train_test_df):
    """
    Plot training vs test performance comparison
    
    Parameters:
    -----------
    train_test_df : pd.DataFrame
        Dataframe with train/test comparison metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    model_names = train_test_df['Model'].tolist()
    train_r2_scores = train_test_df['Train R²'].tolist()
    test_r2_scores = train_test_df['Test R²'].tolist()
    
    x = np.arange(len(model_names))
    width = 0.35
    
    # Plot 1: R² Score Comparison
    bars1 = axes[0].barh(x - width/2, train_r2_scores, width, label='Training R²', color='skyblue')
    bars2 = axes[0].barh(x + width/2, test_r2_scores, width, label='Test R²', color='coral')
    
    axes[0].set_xlabel('R² Score', fontsize=12)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(model_names)
    axes[0].set_title('Training vs Test R² Scores', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (train_val, test_val) in enumerate(zip(train_r2_scores, test_r2_scores)):
        axes[0].text(train_val, i - width/2, f'{train_val:.3f}', ha='left', va='center', fontsize=9)
        axes[0].text(test_val, i + width/2, f'{test_val:.3f}', ha='left', va='center', fontsize=9)
    
    # Plot 2: Accuracy Comparison
    train_acc = train_test_df['Train Accuracy (±10%)'].tolist()
    test_acc = train_test_df['Test Accuracy (±10%)'].tolist()
    
    bars3 = axes[1].barh(x - width/2, train_acc, width, label='Training Accuracy', color='lightgreen')
    bars4 = axes[1].barh(x + width/2, test_acc, width, label='Test Accuracy', color='salmon')
    
    axes[1].set_xlabel('Accuracy (%) - Predictions within ±10%', fontsize=12)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(model_names)
    axes[1].set_title('Training vs Test Accuracy (±10%)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (train_val, test_val) in enumerate(zip(train_acc, test_acc)):
        axes[1].text(train_val, i - width/2, f'{train_val:.1f}%', ha='left', va='center', fontsize=9)
        axes[1].text(test_val, i + width/2, f'{test_val:.1f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('images/03_train_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  [Saved: images/03_train_test_comparison.png]")


def plot_feature_importance(model, feature_cols, model_name):
    """
    Plot feature importance
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_cols : list
        List of feature names
    model_name : str
        Name of the model
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance_df.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='teal')
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{model_name} - Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/04_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  [Saved: images/04_feature_importance.png]")


def plot_actual_vs_predicted(y_test, predictions, model_name):
    """
    Plot actual vs predicted values
    
    Parameters:
    -----------
    y_test : array-like
        True values
    predictions : array-like
        Predicted values
    model_name : str
        Name of the model
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test/10000000, predictions/10000000, 
               alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    plt.plot([y_test.min()/10000000, y_test.max()/10000000], 
            [y_test.min()/10000000, y_test.max()/10000000], 
            'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price (Rs. Crores)', fontsize=12)
    plt.ylabel('Predicted Price (Rs. Crores)', fontsize=12)
    plt.title(f'{model_name} - Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/05_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  [Saved: images/05_actual_vs_predicted.png]")


def plot_residuals(y_test, predictions, model_name):
    """
    Plot residual analysis
    
    Parameters:
    -----------
    y_test : array-like
        True values
    predictions : array-like
        Predicted values
    model_name : str
        Name of the model
    """
    residuals = y_test - predictions
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residual Plot
    axes[0].scatter(predictions/10000000, residuals/10000000, 
                   alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Price (Rs. Crores)', fontsize=12)
    axes[0].set_ylabel('Residuals (Rs. Crores)', fontsize=12)
    axes[0].set_title(f'{model_name} - Residual Plot', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual Distribution
    axes[1].hist(residuals/10000000, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals (Rs. Crores)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/06_residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  [Saved: images/06_residual_analysis.png]")
    
    print(f"Mean Residual: Rs.{residuals.mean()/10000000:.4f} Cr (Should be close to 0)")
    print(f"Std Residual: Rs.{residuals.std()/10000000:.4f} Cr")


def plot_price_distribution(df, save_path='images/07_price_distribution.png'):
    """
    Plot price distribution by locality and overall
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing price data
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall price distribution
    axes[0, 0].hist(df['Price_Clean']/10000000, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Price (Rs. Crores)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Overall Price Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(df['Price_Clean'].mean()/10000000, color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: ₹{df['Price_Clean'].mean()/10000000:.2f} Cr")
    axes[0, 0].axvline(df['Price_Clean'].median()/10000000, color='green', linestyle='--', 
                       linewidth=2, label=f"Median: ₹{df['Price_Clean'].median()/10000000:.2f} Cr")
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Price distribution by BHK
    bhk_order = sorted(df['BHK_Num'].unique())
    df_bhk = df[df['BHK_Num'].isin(bhk_order[:6])]  # Top 6 BHK types
    
    for bhk in bhk_order[:6]:
        subset = df_bhk[df_bhk['BHK_Num'] == bhk]
        axes[0, 1].hist(subset['Price_Clean']/10000000, bins=30, alpha=0.5, 
                       label=f'{int(bhk)} BHK', edgecolor='black')
    
    axes[0, 1].set_xlabel('Price (Rs. Crores)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Price Distribution by BHK Configuration', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Top 10 localities by average price
    if 'Locality_Grouped' in df.columns:
        # Standardize locality variations
        df_locality = df.copy()
        df_locality['Locality_Grouped'] = df_locality['Locality_Grouped'].replace({
            'Science City Road': 'Science City',
            'science city road': 'Science City',
            'science city': 'Science City',
            'Ambli Bopal Road': 'Ambli',
            'ambli bopal road': 'Ambli',
            'Judges Bunglow Road': 'Bodakdev',
            'judges bunglow road': 'Bodakdev'
        })
        
        # Calculate mean price and count per locality
        locality_stats = df_locality.groupby('Locality_Grouped')['Price_Clean'].agg(['mean', 'count'])
        
        # Filter localities with at least 4 properties
        locality_stats = locality_stats[locality_stats['count'] >= 4]
        
        # Get top 10 by mean price
        top_localities = locality_stats['mean'].sort_values(ascending=False).head(10)
        
        axes[1, 0].barh(range(len(top_localities)), top_localities.values/10000000, color='coral')
        axes[1, 0].set_yticks(range(len(top_localities)))
        axes[1, 0].set_yticklabels(top_localities.index)
        axes[1, 0].set_xlabel('Average Price (Rs. Crores)', fontsize=12)
        axes[1, 0].set_title('Top 10 Localities by Average Price (≥4 properties)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Plot 4: Price per sqft distribution
    axes[1, 1].hist(df['Price_Per_SqFt'], bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 1].set_xlabel('Price per Sq.Ft (Rs.)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Price per Sq.Ft Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(df['Price_Per_SqFt'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: ₹{df['Price_Per_SqFt'].mean():.0f}")
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  [Saved: {save_path}]")


def plot_bhk_analysis(df, save_path='images/08_bhk_analysis.png'):
    """
    Plot comprehensive BHK analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing BHK data
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter to common BHK types
    bhk_counts = df['BHK_Num'].value_counts()
    common_bhks = bhk_counts[bhk_counts >= 10].index
    df_filtered = df[df['BHK_Num'].isin(common_bhks)].copy()
    df_filtered['BHK_Num'] = df_filtered['BHK_Num'].astype(int)
    
    # Plot 1: Average price by BHK
    bhk_price = df_filtered.groupby('BHK_Num')['Price_Clean'].mean().sort_index()
    axes[0, 0].plot(bhk_price.index, bhk_price.values/10000000, marker='o', 
                   linewidth=2, markersize=8, color='darkblue')
    axes[0, 0].set_xlabel('BHK Configuration', fontsize=12)
    axes[0, 0].set_ylabel('Average Price (Rs. Crores)', fontsize=12)
    axes[0, 0].set_title('Average Price by BHK Configuration', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average area by BHK
    bhk_area = df_filtered.groupby('BHK_Num')['Area_SqFt'].mean().sort_index()
    axes[0, 1].plot(bhk_area.index, bhk_area.values, marker='s', 
                   linewidth=2, markersize=8, color='darkgreen')
    axes[0, 1].set_xlabel('BHK Configuration', fontsize=12)
    axes[0, 1].set_ylabel('Average Area (Sq.Ft)', fontsize=12)
    axes[0, 1].set_title('Average Area by BHK Configuration', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: BHK distribution
    bhk_dist = df_filtered['BHK_Num'].value_counts().sort_index()
    axes[1, 0].bar(bhk_dist.index, bhk_dist.values, color='teal', edgecolor='black')
    axes[1, 0].set_xlabel('BHK Configuration', fontsize=12)
    axes[1, 0].set_ylabel('Number of Properties', fontsize=12)
    axes[1, 0].set_title('Property Distribution by BHK', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (bhk, count) in enumerate(zip(bhk_dist.index, bhk_dist.values)):
        axes[1, 0].text(bhk, count, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Box plot of price by BHK
    bhk_data = [df_filtered[df_filtered['BHK_Num'] == bhk]['Price_Clean'].values/10000000 
                for bhk in sorted(common_bhks)]
    bp = axes[1, 1].boxplot(bhk_data, labels=sorted(common_bhks), patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    
    axes[1, 1].set_xlabel('BHK Configuration', fontsize=12)
    axes[1, 1].set_ylabel('Price (Rs. Crores)', fontsize=12)
    axes[1, 1].set_title('Price Distribution by BHK (Box Plot)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  [Saved: {save_path}]")


def plot_furnishing_impact(df, save_path='images/09_furnishing_impact.png'):
    """
    Plot furnishing impact on price
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing furnishing data
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average price by furnishing status
    furn_price = df.groupby('Furnishing')['Price_Clean'].mean().sort_values(ascending=False)
    axes[0, 0].barh(range(len(furn_price)), furn_price.values/10000000, color='steelblue', edgecolor='black')
    axes[0, 0].set_yticks(range(len(furn_price)))
    axes[0, 0].set_yticklabels(furn_price.index)
    axes[0, 0].set_xlabel('Average Price (Rs. Crores)', fontsize=12)
    axes[0, 0].set_title('Average Price by Furnishing Status', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(furn_price.values/10000000):
        axes[0, 0].text(val, i, f'₹{val:.2f} Cr', va='center', ha='left', fontsize=10, fontweight='bold')
    
    # Plot 2: Furnishing distribution
    furn_dist = df['Furnishing'].value_counts()
    colors_furn = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700', '#FF99CC']
    axes[0, 1].pie(furn_dist.values, labels=furn_dist.index, autopct='%1.1f%%', 
                   startangle=90, colors=colors_furn[:len(furn_dist)])
    axes[0, 1].set_title('Furnishing Status Distribution', fontsize=14, fontweight='bold')
    
    # Plot 3: Price per sqft by furnishing
    furn_price_sqft = df.groupby('Furnishing')['Price_Per_SqFt'].mean().sort_values(ascending=False)
    axes[1, 0].bar(range(len(furn_price_sqft)), furn_price_sqft.values, color='coral', edgecolor='black')
    axes[1, 0].set_xticks(range(len(furn_price_sqft)))
    axes[1, 0].set_xticklabels(furn_price_sqft.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Average Price per Sq.Ft (Rs.)', fontsize=12)
    axes[1, 0].set_title('Price per Sq.Ft by Furnishing Status', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(furn_price_sqft.values):
        axes[1, 0].text(i, val, f'₹{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Box plot of price by furnishing
    furn_categories = df['Furnishing'].unique()
    furn_data = [df[df['Furnishing'] == furn]['Price_Clean'].values/10000000 
                 for furn in furn_categories]
    bp = axes[1, 1].boxplot(furn_data, labels=furn_categories, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors_furn[:len(furn_categories)]):
        patch.set_facecolor(color)
    
    axes[1, 1].set_xticklabels(furn_categories, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Price (Rs. Crores)', fontsize=12)
    axes[1, 1].set_title('Price Distribution by Furnishing (Box Plot)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  [Saved: {save_path}]")


def plot_outlier_analysis(df_original, df_clean, save_path='images/10_outlier_analysis.png'):
    """
    Plot outlier detection and removal analysis
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataframe before outlier removal
    df_clean : pd.DataFrame
        Dataframe after outlier removal
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Price comparison before/after outlier removal
    axes[0, 0].hist(df_original['Price_Clean']/10000000, bins=50, alpha=0.5, 
                   label='Before Outlier Removal', color='red', edgecolor='black')
    axes[0, 0].hist(df_clean['Price_Clean']/10000000, bins=50, alpha=0.5, 
                   label='After Outlier Removal', color='green', edgecolor='black')
    axes[0, 0].set_xlabel('Price (Rs. Crores)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Price Distribution: Before vs After Outlier Removal', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Area comparison before/after outlier removal
    axes[0, 1].hist(df_original['Area_SqFt'], bins=50, alpha=0.5, 
                   label='Before Outlier Removal', color='red', edgecolor='black')
    axes[0, 1].hist(df_clean['Area_SqFt'], bins=50, alpha=0.5, 
                   label='After Outlier Removal', color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Area (Sq.Ft)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Area Distribution: Before vs After Outlier Removal', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Box plot comparison for price
    data_price = [df_original['Price_Clean']/10000000, df_clean['Price_Clean']/10000000]
    bp1 = axes[1, 0].boxplot(data_price, labels=['Before', 'After'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightcoral')
    bp1['boxes'][1].set_facecolor('lightgreen')
    axes[1, 0].set_ylabel('Price (Rs. Crores)', fontsize=12)
    axes[1, 0].set_title('Price Box Plot: Before vs After Outlier Removal', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Statistics table
    stats_data = {
        'Metric': ['Count', 'Mean Price (Cr)', 'Median Price (Cr)', 'Std Price (Cr)', 
                   'Mean Area (SqFt)', 'Median Area (SqFt)'],
        'Before': [
            len(df_original),
            df_original['Price_Clean'].mean()/10000000,
            df_original['Price_Clean'].median()/10000000,
            df_original['Price_Clean'].std()/10000000,
            df_original['Area_SqFt'].mean(),
            df_original['Area_SqFt'].median()
        ],
        'After': [
            len(df_clean),
            df_clean['Price_Clean'].mean()/10000000,
            df_clean['Price_Clean'].median()/10000000,
            df_clean['Price_Clean'].std()/10000000,
            df_clean['Area_SqFt'].mean(),
            df_clean['Area_SqFt'].median()
        ]
    }
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=[[f"{stats_data['Metric'][i]}", 
                                       f"{stats_data['Before'][i]:.2f}" if i > 0 else f"{int(stats_data['Before'][i])}",
                                       f"{stats_data['After'][i]:.2f}" if i > 0 else f"{int(stats_data['After'][i])}"] 
                                      for i in range(len(stats_data['Metric']))],
                            colLabels=['Metric', 'Before', 'After'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Statistical Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  [Saved: {save_path}]")
    print(f"\nOutliers Removed: {len(df_original) - len(df_clean)} properties ({(len(df_original) - len(df_clean))/len(df_original)*100:.2f}%)")


def plot_top_localities_by_median_price(df, top_n=20, save_path='images/top_localities_mean_price.png'):
    """
    Plot top N localities by mean price (≥4 properties)
    Standardizes locality names (Science City Road → Science City, etc.)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing Locality and Price_Clean columns
    top_n : int
        Number of top localities to display (default: 20)
    save_path : str
        Path to save the plot
    """
    # Create a copy to avoid modifying original data
    df_copy = df.copy()
    
    # Standardize locality variations
    df_copy['Locality'] = df_copy['Locality'].replace({
        'Science City Road': 'Science City',
        'science city road': 'Science City',
        'science city': 'Science City',
        'Ambli Bopal Road': 'Ambli',
        'ambli bopal road': 'Ambli',
        'Judges Bunglow Road': 'Bodakdev',
        'judges bunglow road': 'Bodakdev'
    })
    
    # Calculate mean price per locality
    locality_mean_price = df_copy.groupby('Locality')['Price_Clean'].agg([
        ('Mean_Price', 'mean'),
        ('Count', 'count')
    ]).reset_index()
    
    # Filter localities with at least 4 properties
    locality_mean_price = locality_mean_price[locality_mean_price['Count'] >= 4]
    
    # Sort by mean price and get top N
    top_localities = locality_mean_price.nlargest(top_n, 'Mean_Price')
    
    # Convert to Crores for better readability
    top_localities['Mean_Price_Cr'] = top_localities['Mean_Price'] / 10000000
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_localities)), 
                   top_localities['Mean_Price_Cr'].values,
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_localities))))
    
    # Customize y-axis
    ax.set_yticks(range(len(top_localities)))
    ax.set_yticklabels(top_localities['Locality'].values, fontsize=10)
    
    # Invert y-axis so highest is on top
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (price, count) in enumerate(zip(top_localities['Mean_Price_Cr'].values, 
                                            top_localities['Count'].values)):
        ax.text(price + 0.05, i, f'₹{price:.2f} Cr (n={count})', 
               va='center', fontsize=9)
    
    # Labels and title
    ax.set_xlabel('Mean Price (Rs. Crores)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Locality', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Localities by Mean Property Price (≥4 Properties)\n(Standardized: Science City Road→Science City, Ambli Bopal Road→Ambli, Judges Bunglow Road→Bodakdev)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n  [Saved: {save_path}]")
    print(f"\nTop 3 Localities by Mean Price:")
    for idx, row in top_localities.head(3).iterrows():
        print(f"  {row['Locality']}: ₹{row['Mean_Price_Cr']:.2f} Cr (Properties: {row['Count']})")
    
    return top_localities
