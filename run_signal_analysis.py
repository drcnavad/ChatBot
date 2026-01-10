#!/usr/bin/env python3
"""
Run main_signal_analysis.ipynb
This script executes the main signal analysis notebook
"""

import sys
import os
import subprocess

def run_notebook(notebook_path):
    """Execute a Jupyter notebook using papermill or nbconvert"""
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    print(f"📊 Running notebook: {notebook_path}")
    
    try:
        # Try using papermill first (better for parameterized notebooks)
        result = subprocess.run(
            ['papermill', notebook_path, '-', '--log-output'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print("✅ Notebook executed successfully")
            return True
        else:
            print(f"⚠️ Papermill failed, trying nbconvert...")
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("⚠️ Papermill not found, trying nbconvert...")
    except subprocess.TimeoutExpired:
        print("❌ Notebook execution timed out")
        return False
    except Exception as e:
        print(f"⚠️ Error with papermill: {e}, trying nbconvert...")
    
    # Fallback to nbconvert
    try:
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', notebook_path],
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode == 0:
            print("✅ Notebook executed successfully with nbconvert")
            return True
        else:
            print(f"❌ Notebook execution failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Neither papermill nor jupyter nbconvert found. Please install one:")
        print("   pip install papermill")
        print("   or")
        print("   pip install jupyter")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Notebook execution timed out")
        return False
    except Exception as e:
        print(f"❌ Error executing notebook: {e}")
        return False

if __name__ == "__main__":
    notebook_path = os.path.join(os.path.dirname(__file__), "main_signal_analysis.ipynb")
    success = run_notebook(notebook_path)
    sys.exit(0 if success else 1)
