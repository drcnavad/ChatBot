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
    # Get notebook path - try multiple locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    
    # In GitHub Actions, the workspace is the repository root
    # Try repository root first (where the file should be)
    github_workspace = os.getenv("GITHUB_WORKSPACE", cwd)
    
    # Try different paths
    possible_paths = [
        os.path.join(github_workspace, "main_signal_analysis.ipynb"),  # GitHub Actions workspace root
        os.path.join(cwd, "main_signal_analysis.ipynb"),  # Current working directory
        os.path.join(script_dir, "main_signal_analysis.ipynb"),  # Script directory
        "main_signal_analysis.ipynb",  # Relative to current directory
    ]
    
    # Debug output
    print(f"GitHub workspace (GITHUB_WORKSPACE env): {github_workspace}")
    print(f"Current working directory: {cwd}")
    print(f"Script directory: {script_dir}")
    print(f"Looking for notebook in:")
    for path in possible_paths:
        exists = os.path.exists(path)
        abs_path = os.path.abspath(path) if os.path.exists(path) else path
        print(f"  {abs_path} - {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    
    notebook_path = None
    for path in possible_paths:
        if os.path.exists(path):
            notebook_path = os.path.abspath(path)
            print(f"\n✅ Found notebook at: {notebook_path}")
            break
    
    if not notebook_path:
        print(f"\n❌ Notebook not found. Tried all paths above.")
        print(f"\nFiles in current directory ({cwd}):")
        try:
            files = os.listdir(cwd)
            for f in sorted(files)[:30]:
                file_path = os.path.join(cwd, f)
                file_type = "DIR" if os.path.isdir(file_path) else "FILE"
                print(f"  [{file_type}] {f}")
        except Exception as e:
            print(f"  Error listing directory: {e}")
        sys.exit(1)
    
    success = run_notebook(notebook_path)
    sys.exit(0 if success else 1)
