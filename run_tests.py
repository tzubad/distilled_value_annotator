#!/usr/bin/env python3
"""
Simple test runner script.

Usage:
    python run_tests.py              # Run all unit tests
    python run_tests.py --all        # Run all tests including integration
    python run_tests.py --coverage   # Run with coverage report
"""

import sys
import subprocess
import argparse


def run_tests(include_integration=False, coverage=False):
    """Run the test suite."""
    
    # Base pytest command
    cmd = ['python', '-m', 'pytest', 'tests/']
    
    # Add markers to exclude integration tests by default
    if not include_integration:
        cmd.extend(['-m', 'not integration'])
        print("Running unit tests only (use --all to include integration tests)")
    else:
        print("Running all tests (including integration tests)")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(['--cov=.', '--cov-report=html', '--cov-report=term'])
        print("Coverage report will be generated")
    
    print()
    print("Command:", ' '.join(cmd))
    print("=" * 60)
    print()
    
    # Run pytest
    result = subprocess.run(cmd)
    
    if coverage and result.returncode == 0:
        print()
        print("=" * 60)
        print("Coverage report generated in htmlcov/index.html")
    
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run test suite')
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests including integration tests'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    args = parser.parse_args()
    
    return run_tests(
        include_integration=args.all,
        coverage=args.coverage
    )


if __name__ == '__main__':
    sys.exit(main())
