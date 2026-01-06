#!/usr/bin/env python3
"""
Model Evaluation CLI - Run evaluation of annotation models against ground truth.

This script provides a command-line interface for the Model Evaluation Module,
coordinating the evaluation of LLM and MLM models against human-annotated ground truth.

Usage:
    python run_evaluation.py --config evaluation_config.yaml
    python run_evaluation.py --config config.json --verbose
    python run_evaluation.py --config config.yaml --dry-run

Configuration:
    The configuration file (YAML or JSON) should include:
    - ground_truth_path: Path to ground truth CSV/JSON file
    - scripts_path: Path to video scripts directory
    - output_dir: Directory for evaluation reports
    - models: List of model configurations to evaluate

For more information, see the README.md file.
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

from evaluation import (
    EvaluationOrchestrator,
    EvaluationConfigLoader,
    ConfigValidationError,
    EvaluationSummary,
)
from evaluation.adapters import GeminiAdapter, MLMAdapter


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Model Evaluation CLI - Evaluate annotation models against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config evaluation_config.yaml
  %(prog)s --config config.json --verbose
  %(prog)s --config config.yaml --output-dir ./results
  %(prog)s --config config.yaml --dry-run

Configuration File:
  The configuration file should be in YAML or JSON format and include:
  - ground_truth_path: Path to ground truth dataset
  - scripts_path: Path to video scripts
  - output_dir: Output directory for reports
  - models: List of model configurations

For more information, see the README.md file.
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to the evaluation configuration file (YAML or JSON format)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Override output directory from config file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output (only show errors and final summary)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running evaluation'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Run evaluation only for specified models (by name)'
    )
    
    parser.add_argument(
        '--skip-reports',
        action='store_true',
        help='Skip report generation (useful for quick metrics check)'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """
    Configure logging for the evaluation CLI.
    
    Args:
        verbose: Enable DEBUG level logging
        quiet: Suppress INFO level, only show warnings and errors
        
    Returns:
        Configured logger instance
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger('EvaluationCLI')


def register_default_adapters():
    """Register the default model adapters with the orchestrator."""
    EvaluationOrchestrator.register_adapter("GeminiAdapter", GeminiAdapter)
    EvaluationOrchestrator.register_adapter("MLMAdapter", MLMAdapter)


def print_banner():
    """Print the evaluation CLI banner."""
    print()
    print("=" * 70)
    print("  MODEL EVALUATION MODULE")
    print("  Evaluating annotation models against ground truth")
    print("=" * 70)
    print()


def print_config_summary(config, logger: logging.Logger):
    """Print a summary of the loaded configuration."""
    logger.info("Configuration Summary:")
    logger.info(f"  Ground Truth: {config.ground_truth_path}")
    logger.info(f"  Scripts Path: {config.scripts_path}")
    logger.info(f"  Output Dir:   {config.output_dir}")
    logger.info(f"  Models:       {len(config.models)}")
    for model in config.models:
        logger.info(f"    - {model.model_name} ({model.adapter_class})")
    if config.sample_size:
        logger.info(f"  Sample Size:  {config.sample_size}")
    if config.random_seed:
        logger.info(f"  Random Seed:  {config.random_seed}")


def print_progress(stage: str, current: int, total: int, model_name: Optional[str] = None):
    """Print progress indicator."""
    if model_name:
        print(f"\r  [{stage}] {model_name}: {current}/{total}", end="", flush=True)
    else:
        print(f"\r  [{stage}] {current}/{total}", end="", flush=True)


def print_summary(summary: EvaluationSummary):
    """Print the evaluation summary."""
    print()
    print("=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print()
    
    print(f"  Models Configured:    {summary.total_models}")
    print(f"  Models Successful:    {summary.successful_models}")
    print(f"  Models Failed:        {summary.failed_models}")
    print()
    
    print(f"  Videos in Dataset:    {summary.total_videos}")
    print(f"  Predictions Made:     {summary.total_predictions}")
    print(f"  Successful:           {summary.successful_predictions}")
    print(f"  Failed:               {summary.failed_predictions}")
    print()
    
    if summary.total_predictions > 0:
        success_rate = (summary.successful_predictions / summary.total_predictions) * 100
        print(f"  Overall Success Rate: {success_rate:.1f}%")
        print()
    
    print(f"  Execution Time:       {summary.elapsed_time:.2f} seconds")
    print()
    
    if summary.reports_generated:
        print("  Reports Generated:")
        for report in summary.reports_generated:
            print(f"    - {report}")
        print()
    
    if summary.adapter_errors:
        print("  Adapter Errors:")
        for model_name, error in summary.adapter_errors.items():
            print(f"    - {model_name}: {error}")
        print()
    
    print("=" * 70)


def run_evaluation(args: argparse.Namespace, logger: logging.Logger) -> int:
    """
    Run the evaluation workflow.
    
    Args:
        args: Parsed command-line arguments
        logger: Logger instance
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    start_time = time.time()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    # Load and validate configuration
    logger.info(f"Loading configuration from: {args.config}")
    try:
        loader = EvaluationConfigLoader()
        config = loader.load(str(config_path))
        logger.info("Configuration loaded and validated successfully")
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
        logger.info(f"Output directory overridden to: {args.output_dir}")
    
    # Print configuration summary
    print_config_summary(config, logger)
    
    # Dry run - just validate and exit
    if args.dry_run:
        logger.info("Dry run complete - configuration is valid")
        print("\n✓ Configuration validated successfully (dry run)")
        return 0
    
    # Register default adapters
    register_default_adapters()
    
    # Create orchestrator
    logger.info("Initializing evaluation orchestrator...")
    try:
        orchestrator = EvaluationOrchestrator(config=config)
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        return 1
    
    # Load ground truth
    logger.info("Loading ground truth dataset...")
    try:
        ground_truth = orchestrator.load_ground_truth()
        logger.info(f"Loaded {ground_truth.valid_count} valid videos from ground truth")
    except Exception as e:
        logger.error(f"Failed to load ground truth: {e}")
        return 1
    
    # Initialize adapters
    logger.info("Initializing model adapters...")
    try:
        init_results = orchestrator.initialize_adapters()
        successful = sum(1 for r in init_results if r.success)
        logger.info(f"Initialized {successful}/{len(init_results)} adapters successfully")
        
        for result in init_results:
            if result.success:
                logger.debug(f"  ✓ {result.model_name}")
            else:
                logger.warning(f"  ✗ {result.model_name}: {result.error_message}")
    except Exception as e:
        logger.error(f"Failed to initialize adapters: {e}")
        return 1
    
    # Check if any adapters initialized
    if not orchestrator._adapters:
        logger.error("No adapters initialized successfully - cannot proceed")
        return 1
    
    # Run predictions
    logger.info("Running predictions...")
    try:
        storage = orchestrator.run_predictions()
        
        # Get prediction counts
        counts = orchestrator.get_prediction_counts()
        for model_name, model_counts in counts.items():
            logger.info(
                f"  {model_name}: {model_counts['success']}/{model_counts['total']} successful"
            )
    except Exception as e:
        logger.error(f"Failed during prediction phase: {e}")
        return 1
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    try:
        model_results = orchestrator.calculate_metrics()
        for model_name, result in model_results.items():
            logger.info(
                f"  {model_name}: Macro F1 = {result.endorsed_aggregate.macro_f1:.4f}"
            )
    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        return 1
    
    # Generate reports (unless skipped)
    if not args.skip_reports:
        logger.info("Generating reports...")
        try:
            reports = orchestrator.generate_reports(model_results)
            for report_path in reports:
                logger.info(f"  Generated: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            return 1
    else:
        logger.info("Skipping report generation (--skip-reports)")
    
    # Create summary
    elapsed_time = time.time() - start_time
    
    # Build summary manually since we didn't use run()
    total_predictions = sum(c['total'] for c in counts.values())
    successful_predictions = sum(c['success'] for c in counts.values())
    
    summary = EvaluationSummary(
        total_models=len(config.models),
        successful_models=len(orchestrator._adapters),
        failed_models=len(config.models) - len(orchestrator._adapters),
        total_videos=ground_truth.valid_count,
        total_predictions=total_predictions,
        successful_predictions=successful_predictions,
        failed_predictions=total_predictions - successful_predictions,
        elapsed_time=elapsed_time,
        generated_reports={} if args.skip_reports else {name: Path(path) for name, path in reports.items()},
        model_errors=orchestrator.adapter_errors,
    )
    
    # Print summary
    if not args.quiet:
        print_summary(summary)
    
    # Return success if we have results
    if summary.successful_predictions > 0:
        logger.info("Evaluation completed successfully!")
        return 0
    else:
        logger.warning("Evaluation completed but no successful predictions")
        return 1


def main() -> int:
    """
    Main entry point for the evaluation CLI.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        logger = setup_logging(verbose=args.verbose, quiet=args.quiet)
        
        # Print banner (unless quiet)
        if not args.quiet:
            print_banner()
        
        # Run evaluation
        return run_evaluation(args, logger)
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        return 130
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
