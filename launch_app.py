"""
Streamlit App Launcher
Launch script for the Real Estate Price Prediction Streamlit app
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        
        print("\n💡 To install missing packages, run:")
        print("   pip install -r requirements_streamlit.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True


def check_files():
    """Check if required files exist"""
    required_files = [
        'app.py',
        'streamlit_utils.py',
        'streamlit_config.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files are present!")
    return True


def check_data_and_models():
    """Check for data and model files"""
    data_files = ['data/cleaned_data.csv', 'data/ahmedabad_real_estate_data.csv']
    model_files = [
        'models/best_price_prediction_model.pkl',
        'models/feature_scaler.pkl',
        'models/furnishing_encoder.pkl',
        'models/locality_encoder.pkl'
    ]
    
    data_exists = any(os.path.exists(f) for f in data_files)
    models_exist = all(os.path.exists(f) for f in model_files)
    
    if not data_exists:
        print("⚠️  No data files found - app will use sample data")
    else:
        print("✅ Data files found!")
    
    if not models_exist:
        print("⚠️  Some model files missing - app will use dummy models")
    else:
        print("✅ Model files found!")
    
    return True  # Always return True as app can run with sample/dummy data


def launch_app():
    """Launch the Streamlit app"""
    print("\n" + "="*60)
    print("🚀 LAUNCHING REAL ESTATE PRICE PREDICTION APP")
    print("="*60)
    
    # Check system requirements
    print("\n🔍 Checking system requirements...")
    
    if not check_requirements():
        return False
    
    if not check_files():
        return False
    
    check_data_and_models()
    
    print("\n✅ All checks passed! Starting Streamlit app...")
    print("\n" + "="*60)
    print("📱 APP INFORMATION")
    print("="*60)
    print("🌐 URL: http://localhost:8501")
    print("🛑 Stop app: Press Ctrl+C in terminal")
    print("🔄 Reload: Save any file to auto-reload")
    print("="*60)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port=8501',
            '--server.headless=false',
            '--browser.serverAddress=localhost'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        return False
    
    return True


def create_desktop_shortcut():
    """Create a desktop shortcut (Windows only)"""
    if sys.platform == "win32":
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            path = os.path.join(desktop, "Real Estate Price Predictor.lnk")
            target = sys.executable
            wDir = os.getcwd()
            arguments = f'"{__file__}"'
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(path)
            shortcut.Targetpath = target
            shortcut.Arguments = arguments
            shortcut.WorkingDirectory = wDir
            shortcut.IconLocation = target
            shortcut.save()
            
            print(f"✅ Desktop shortcut created: {path}")
            
        except ImportError:
            print("⚠️  winshell not available - cannot create desktop shortcut")
        except Exception as e:
            print(f"⚠️  Could not create desktop shortcut: {e}")


def show_help():
    """Show help information"""
    print("""
🏠 Real Estate Price Prediction App - Launcher

USAGE:
    python launch_app.py [OPTIONS]

OPTIONS:
    --help, -h          Show this help message
    --check, -c         Check system requirements only
    --shortcut, -s      Create desktop shortcut (Windows only)
    --install, -i       Install required packages

EXAMPLES:
    python launch_app.py           # Launch the app
    python launch_app.py --check   # Check requirements
    python launch_app.py --install # Install packages

TROUBLESHOOTING:
    1. If packages are missing: pip install -r requirements_streamlit.txt
    2. If models are missing: Run main.py first to train models
    3. If data is missing: App will use sample data automatically
    4. If port 8501 is busy: Close other Streamlit apps or use different port
    
For more information, visit: https://streamlit.io/
    """)


def install_packages():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        if os.path.exists('requirements_streamlit.txt'):
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '-r', 'requirements_streamlit.txt'
            ])
            print("✅ Packages installed successfully!")
            return True
        else:
            print("❌ requirements_streamlit.txt not found")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def main():
    """Main function"""
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h']:
            show_help()
        elif arg in ['--check', '-c']:
            print("🔍 Checking system requirements...")
            check_requirements()
            check_files()
            check_data_and_models()
        elif arg in ['--shortcut', '-s']:
            create_desktop_shortcut()
        elif arg in ['--install', '-i']:
            install_packages()
        else:
            print(f"❌ Unknown argument: {arg}")
            print("Use --help for available options")
    else:
        # Default action: launch app
        success = launch_app()
        if not success:
            print("\n💡 Try running with --help for troubleshooting tips")


if __name__ == "__main__":
    main()