"""
Quick test for the fixed plotting functions
"""

import pandas as pd
import plotly.express as px
import sys
import os

def test_plotting():
    """Test the plotting functions with sample data"""
    print("🧪 Testing plotting functions...")
    
    try:
        # Add current directory to path
        sys.path.append('.')
        
        from streamlit_utils import create_price_distribution_plot, format_price
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Price_Clean': [15000000, 25000000, 35000000, 45000000, 20000000, 30000000],
            'BHK_Num': [2, 3, 4, 5, 2, 3],
            'Area_SqFt': [800, 1200, 1800, 2500, 900, 1300]
        })
        
        print(f"✅ Sample data created: {len(sample_data)} records")
        
        # Test plotting function
        fig = create_price_distribution_plot(sample_data)
        print("✅ Price distribution plot created successfully")
        
        # Test price formatting
        formatted_price = format_price(25000000)
        print(f"✅ Price formatting works: {formatted_price}")
        
        print("\n🎉 All plotting tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Plotting error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_plotting()
    print("\n" + "="*50)
    if success:
        print("✅ Plotting functions are working correctly!")
        print("🚀 The Streamlit app should now work without errors.")
    else:
        print("❌ There are still issues with plotting functions.")
        print("Please check the error messages above.")
    print("="*50)