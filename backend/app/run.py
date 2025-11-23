#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🚀 Script Setup - Place this in the root directory (backend)
Run the application from anywhere!
"""

import sys
from pathlib import Path

# ==============================================================================

def check_structure():
    """Check the required directory structure (files must be inside 'app')"""
    print("=" * 80)
    print("📋 Checking Project Structure (Files inside 'app' folder)")
    print("=" * 80)
    
    current_dir = Path.cwd()
    app_dir = current_dir / 'app' # ✅ The required subdirectory
    print(f"📁 Current Directory: {current_dir}")
    
    # 1. Check if the 'app' directory exists
    if not app_dir.is_dir():
        print(f"❌ 'app' directory missing in: {current_dir}")
        print("⚠️  Ensure you are running run.py from the parent directory (backend).")
        return False
        
    required_files_in_app = ['main.py', 'ai_comparator.py', 'utils.py']
    missing_files = []
    
    # 2. Check for required files inside 'app'
    for file in required_files_in_app:
        file_path = app_dir / file
        if file_path.exists():
            print(f"✅ app/{file} found")
        else:
            missing_files.append(file)
            print(f"❌ app/{file} missing")
    
    print()
    
    if missing_files:
        print(f"⚠️  Missing files inside the 'app' folder: {', '.join(missing_files)}")
        return False
    
    return True

# ==============================================================================

def run_server():
    """Run the server using the correct module path"""
    print("\n" + "=" * 80)
    print("🚀 Starting Application Server")
    print("=" * 80 + "\n")
    
    try:
        # uvicorn and fastapi must be installed first
        import uvicorn
        
        # ✅ CRITICAL CHANGE: Using the correct module path 'app.main:app'
        print("💡 Execution Command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        print("⏳ Running server...\n")
        
        uvicorn.run(
            "app.main:app", # Points to main.py inside the 'app' folder
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    
    except ImportError:
        print("❌ Error: uvicorn library is missing.")
        print("Please install using: pip install uvicorn")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during startup: {e}")
        return False
    
    return True

# ==============================================================================

if __name__ == "__main__":
    # Check structure first
    if not check_structure():
        print("\n⚠️  Please ensure all files are located inside the backend/app/ folder!")
        sys.exit(1)
    
    # Run the server
    if not run_server():
        sys.exit(1)