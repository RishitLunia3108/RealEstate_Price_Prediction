"""
Main Script - Ahmedabad Real Estate Price Prediction
Complete automated pipeline: Scraping → Cleaning → Training → Visualization
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from data_cleaning import clean_scraped_data
from data_preprocessing import preprocess_pipeline
from model_training import (
    split_data, scale_features, train_all_models,
    calculate_train_test_comparison, get_best_model,
    calculate_additional_metrics
)
from visualization import (
    plot_correlation_matrix, plot_model_comparison,
    plot_train_test_comparison, plot_feature_importance,
    plot_actual_vs_predicted, plot_residuals,
    plot_price_distribution, plot_bhk_analysis,
    plot_furnishing_impact, plot_outlier_analysis
)
from model_utils import (
    save_model, create_comparison_dataframe,
    print_model_summary, print_train_test_summary,
    create_sample_predictions, print_insights
)


def run_scraper():
    """
    Run web scraper to collect real estate data
    """
    print("\n" + "="*80)
    print("STEP 1: DATA COLLECTION (WEB SCRAPING)")
    print("="*80)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    raw_data_file = 'data/ahmedabad_real_estate_data.csv'
    
    # Check if data already exists
    if os.path.exists(raw_data_file):
        print(f"\n✓ Raw data file already exists: {raw_data_file}")
        response = input("\nDo you want to scrape new data? (y/n): ").lower()
        if response != 'y':
            print("Skipping data scraping, using existing file...")
            return raw_data_file
    
    print("\nStarting web scraper...")
    try:
        import scraper
        scraper.main()
        print(f"\n✓ Scraping completed successfully!")
        return raw_data_file
    except Exception as e:
        print(f"\n✗ Error during scraping: {e}")
        if os.path.exists(raw_data_file):
            print(f"Using existing data file: {raw_data_file}")
            return raw_data_file
        else:
            raise


def clean_data():
    """
    Clean scraped data and prepare for modeling
    """
    print("\n" + "="*80)
    print("STEP 2: DATA CLEANING")
    print("="*80)
    
    cleaned_data_file = 'data/cleaned_data.csv'
    
    # Check if cleaned data already exists
    if os.path.exists(cleaned_data_file):
        print(f"\n✓ Cleaned data file already exists: {cleaned_data_file}")
        response = input("\nDo you want to clean data again? (y/n): ").lower()
        if response != 'y':
            print("Skipping data cleaning, using existing file...")
            return cleaned_data_file
    
    print("\nStarting data cleaning pipeline...")
    clean_scraped_data(
        input_file='data/ahmedabad_real_estate_data.csv',
        output_file=cleaned_data_file
    )
    print(f"\n✓ Data cleaning completed successfully!")
    return cleaned_data_file


def main():
    """Main execution function"""
    
    print("="*80)
    print("AHMEDABAD REAL ESTATE PRICE PREDICTION MODEL")
    print("AUTOMATED END-TO-END PIPELINE")
    print("="*80)
    
    # STEP 1: Scrape Data
    raw_data_file = run_scraper()
    
    # STEP 2: Clean Data
    cleaned_data_file = clean_data()
    
    # STEP 3: Preprocessing
    print("\n" + "="*80)
    print("STEP 3: DATA PREPROCESSING")
    print("="*80)
    preprocessed_data = preprocess_pipeline(
        filepath=cleaned_data_file,
        percentile=0.99,
        min_properties=5
    )
    
    X = preprocessed_data['X']
    y = preprocessed_data['y']
    df = preprocessed_data['df']
    df_original = preprocessed_data['df_before_outliers']
    feature_cols = preprocessed_data['feature_cols']
    le_furn = preprocessed_data['le_furn']
    le_loc = preprocessed_data['le_loc']
    le_combo = preprocessed_data['le_combo']
    
    # STEP 3A: Enhanced Data Visualizations
    print("\n" + "="*80)
    print("STEP 3A: EXPLORATORY DATA ANALYSIS VISUALIZATIONS")
    print("="*80)
    
    # Price distribution analysis
    print("\nGenerating price distribution plots...")
    plot_price_distribution(df)
    
    # BHK analysis
    print("\nGenerating BHK analysis plots...")
    plot_bhk_analysis(df)
    
    # Furnishing impact
    print("\nGenerating furnishing impact plots...")
    plot_furnishing_impact(df)
    
    # Outlier analysis
    print("\nGenerating outlier analysis plots...")
    plot_outlier_analysis(df_original, df)
    
    # Visualize correlations
    print("\nFeature Correlations:")
    plot_correlation_matrix(df, ['Price_Clean'] + feature_cols)
    
    # STEP 4: Train-Test Split and Scaling
    print("\n" + "="*80)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("="*80)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # STEP 5: Model Training
    print("\n" + "="*80)
    print("STEP 5: MODEL TRAINING")
    print("="*80)
    results = train_all_models(X_train, X_test, y_train, y_test, 
                               X_train_scaled, X_test_scaled)
    
    # STEP 6: Model Comparison
    print("\n" + "="*80)
    print("STEP 6: MODEL COMPARISON")
    print("="*80)
    comparison_df = create_comparison_dataframe(results)
    print(comparison_df.to_string(index=False))
    
    # Visualize comparison
    plot_model_comparison(comparison_df)
    
    # STEP 7: Training vs Test Comparison
    print("\n" + "="*80)
    print("STEP 7: TRAINING vs TEST COMPARISON")
    print("="*80)
    train_test_comparison = calculate_train_test_comparison(
        results, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled
    )
    
    train_test_df = print_train_test_summary(train_test_comparison)
    
    # Visualize train/test comparison
    plot_train_test_comparison(train_test_df)
    
    # STEP 8: Best Model Analysis
    print("\n" + "="*80)
    print("STEP 8: BEST MODEL ANALYSIS")
    print("="*80)
    best_model_name, best_model_info = get_best_model(results)
    
    print(f"Best Model: {best_model_name}")
    print(f"R² Score: {best_model_info['R2']:.4f}")
    
    # Feature importance
    plot_feature_importance(best_model_info['model'], feature_cols, best_model_name)
    
    # Actual vs Predicted
    plot_actual_vs_predicted(y_test, best_model_info['predictions'], best_model_name)
    
    # Residual analysis
    plot_residuals(y_test, best_model_info['predictions'], best_model_name)
    
    # STEP 9: Performance Summary
    print("\n" + "="*80)
    print("STEP 9: PERFORMANCE SUMMARY")
    print("="*80)
    additional_metrics = calculate_additional_metrics(y_test, best_model_info['predictions'])
    
    print_model_summary(
        best_model_name, best_model_info, additional_metrics,
        len(df), len(X_train), len(X_test)
    )
    
    # Sample predictions
    print("\nSample Predictions (First 15 test properties):")
    sample_predictions = create_sample_predictions(y_test, best_model_info['predictions'])
    print(sample_predictions.to_string(index=False))
    
    # STEP 10: Insights
    print("\n" + "="*80)
    print("STEP 10: KEY INSIGHTS")
    print("="*80)
    print_insights(best_model_info, feature_cols, 
                   additional_metrics['Accuracy_20pct'], len(df))
    
    # STEP 11: Save Model
    print("\n" + "="*80)
    print("STEP 11: SAVE MODEL")
    print("="*80)
    save_model(best_model_info['model'], scaler, le_furn, le_loc, le_combo)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\n✓ Scraped data saved to: data/ahmedabad_real_estate_data.csv")
    print("✓ Cleaned data saved to: data/cleaned_data.csv")
    print("✓ Models saved to: models/")
    print("✓ Visualizations saved to: images/")
    print("\n📊 VISUALIZATIONS GENERATED:")
    print("  01_correlation_matrix.png      - Feature correlations")
    print("  02_model_comparison.png        - Model performance comparison")
    print("  03_train_test_comparison.png   - Training vs test performance")
    print("  04_feature_importance.png      - Feature importance analysis")
    print("  05_actual_vs_predicted.png     - Prediction accuracy scatter plot")
    print("  06_residual_analysis.png       - Residual analysis")
    print("  07_price_distribution.png      - Price distribution analysis")
    print("  08_bhk_analysis.png            - BHK configuration analysis")
    print("  09_furnishing_impact.png       - Furnishing impact on price")
    print("  10_outlier_analysis.png        - Outlier detection analysis")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

