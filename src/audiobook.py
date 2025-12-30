"""
Alias module to allow running 'python -m audiobook'
Redirects to audiobook_creator.cli.main
"""
import sys
from audiobook_creator.cli import main

if __name__ == "__main__":
    sys.exit(main())
