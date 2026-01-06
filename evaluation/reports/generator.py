# Report generator for model evaluation results

import csv
import json
import logging
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.metrics.calculator import (
    ANNOTATION_CATEGORIES,
    ModelEvaluationResult,
    CategoryResult,
    AggregateResult,
)


class ReportGenerator:
    """
    Generates evaluation reports in multiple formats.
    
    Supports:
    - CSV reports for per-category and aggregate metrics
    - JSON reports with full structured data
    - Comparison reports for multiple models
    - GCS upload support for cloud storage
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory path where reports will be saved.
                       Can be a local path or GCS URI (gs://bucket/path).
                       Directory will be created if it doesn't exist.
        """
        self._output_dir_str = output_dir
        self._is_gcs = output_dir.startswith("gs://")
        
        if self._is_gcs:
            # For GCS, use a temporary local directory
            self._output_dir = Path(tempfile.mkdtemp(prefix="evaluation_reports_"))
            self._gcs_uri = output_dir
            logging.info(f"ReportGenerator initialized with GCS output: {self._gcs_uri}")
            logging.info(f"Using temporary local directory: {self._output_dir}")
        else:
            self._output_dir = Path(output_dir)
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._gcs_uri = None
            logging.info(f"ReportGenerator initialized with output directory: {self._output_dir}")
    
    @property
    def output_dir(self) -> Path:
        """Get the output directory path."""
        return self._output_dir
    
    def _upload_to_gcs(self, local_path: Path) -> str:
        """
        Upload a file to GCS if GCS output is configured.
        
        Args:
            local_path: Path to the local file to upload
            
        Returns:
            GCS URI of the uploaded file, or local path string if not using GCS
        """
        if not self._is_gcs:
            return str(local_path)
        
        try:
            from google.cloud import storage
            import os
            
            # Parse GCS URI
            gcs_path = self._gcs_uri.replace("gs://", "")
            bucket_name = gcs_path.split("/")[0]
            base_path = "/".join(gcs_path.split("/")[1:]) if "/" in gcs_path else ""
            
            # Initialize GCS client
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
            if project_id:
                client = storage.Client(project=project_id)
            else:
                client = storage.Client()
            
            bucket = client.bucket(bucket_name)
            
            # Construct destination path
            file_name = local_path.name
            if base_path:
                gcs_file_path = f"{base_path.rstrip('/')}/{file_name}"
            else:
                gcs_file_path = file_name
            
            # Upload file
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(str(local_path))
            
            gcs_uri = f"gs://{bucket_name}/{gcs_file_path}"
            logging.info(f"Uploaded {local_path.name} to {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logging.error(f"Failed to upload {local_path} to GCS: {e}")
            return str(local_path)
    
    def generate_all_reports(
        self,
        results: List[ModelEvaluationResult],
        timestamp: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Generate all report formats for one or more model evaluation results.
        
        Args:
            results: List of model evaluation results to report
            timestamp: Optional timestamp for report filenames (default: current time)
            
        Returns:
            Dictionary mapping report type to file path
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        generated_files = {}
        
        for result in results:
            # Generate per-model reports
            csv_path = self.generate_csv_report(result, timestamp)
            generated_files[f"{result.model_name}_csv"] = csv_path
            
            json_path = self.generate_json_report(result, timestamp)
            generated_files[f"{result.model_name}_json"] = json_path
        
        # Generate comparison report if multiple models
        if len(results) > 1:
            comparison_path = self.generate_comparison_report(results, timestamp)
            generated_files["comparison"] = comparison_path
        
        logging.info(f"Generated {len(generated_files)} report files")
        return generated_files
    
    def generate_csv_report(
        self,
        result: ModelEvaluationResult,
        timestamp: Optional[str] = None,
    ) -> Path:
        """
        Generate CSV report for a single model's evaluation results.
        
        Creates two CSV files:
        - Per-category metrics CSV with all 19 categories
        - Aggregate metrics CSV with macro/weighted F1 scores
        
        Args:
            result: Model evaluation result to report
            timestamp: Optional timestamp for filename
            
        Returns:
            Path to the per-category metrics CSV file (or GCS URI if using GCS)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_name_safe = self._sanitize_filename(result.model_name)
        
        # Generate per-category metrics CSV
        category_csv_path = self._output_dir / f"{model_name_safe}_category_metrics_{timestamp}.csv"
        self._write_category_csv(result, category_csv_path)
        
        # Generate aggregate metrics CSV
        aggregate_csv_path = self._output_dir / f"{model_name_safe}_aggregate_metrics_{timestamp}.csv"
        self._write_aggregate_csv(result, aggregate_csv_path)
        
        # Upload to GCS if configured
        category_uri = self._upload_to_gcs(category_csv_path)
        aggregate_uri = self._upload_to_gcs(aggregate_csv_path)
        
        logging.info(f"Generated CSV reports for {result.model_name}")
        return category_uri if self._is_gcs else category_csv_path
    
    def _write_category_csv(self, result: ModelEvaluationResult, path: Path) -> None:
        """Write per-category metrics to CSV."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'category',
                'endorsed_precision', 'endorsed_recall', 'endorsed_f1', 'endorsed_support',
                'conflict_precision', 'conflict_recall', 'conflict_f1', 'conflict_support',
                'combined_precision', 'combined_recall', 'combined_f1', 'combined_support',
            ])
            
            # Data rows for each category
            for category in ANNOTATION_CATEGORIES:
                endorsed = result.per_category_endorsed.get(category)
                conflict = result.per_category_conflict.get(category)
                combined = result.per_category_combined.get(category)
                
                row = [category]
                
                # Endorsed metrics
                if endorsed:
                    row.extend([endorsed.precision, endorsed.recall, endorsed.f1, endorsed.support])
                else:
                    row.extend([0.0, 0.0, 0.0, 0])
                
                # Conflict metrics
                if conflict:
                    row.extend([conflict.precision, conflict.recall, conflict.f1, conflict.support])
                else:
                    row.extend([0.0, 0.0, 0.0, 0])
                
                # Combined metrics
                if combined:
                    row.extend([combined.precision, combined.recall, combined.f1, combined.support])
                else:
                    row.extend([0.0, 0.0, 0.0, 0])
                
                writer.writerow(row)
    
    def _write_aggregate_csv(self, result: ModelEvaluationResult, path: Path) -> None:
        """Write aggregate metrics to CSV."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'metric_type',
                'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall',
                'categories_evaluated',
            ])
            
            # Endorsed aggregate
            writer.writerow([
                'endorsed',
                result.endorsed_aggregate.macro_f1,
                result.endorsed_aggregate.weighted_f1,
                result.endorsed_aggregate.macro_precision,
                result.endorsed_aggregate.macro_recall,
                result.endorsed_aggregate.categories_evaluated,
            ])
            
            # Conflict aggregate
            writer.writerow([
                'conflict',
                result.conflict_aggregate.macro_f1,
                result.conflict_aggregate.weighted_f1,
                result.conflict_aggregate.macro_precision,
                result.conflict_aggregate.macro_recall,
                result.conflict_aggregate.categories_evaluated,
            ])
            
            # Combined aggregate
            writer.writerow([
                'combined',
                result.combined_aggregate.macro_f1,
                result.combined_aggregate.weighted_f1,
                result.combined_aggregate.macro_precision,
                result.combined_aggregate.macro_recall,
                result.combined_aggregate.categories_evaluated,
            ])
    
    def generate_json_report(
        self,
        result: ModelEvaluationResult,
        timestamp: Optional[str] = None,
    ) -> Path:
        """
        Generate JSON report for a single model's evaluation results.
        
        Exports the full ModelEvaluationResult as structured JSON including:
        - Model metadata (name, prediction counts)
        - Per-category metrics for all value types
        - Aggregate metrics for all value types
        
        Args:
            result: Model evaluation result to report
            timestamp: Optional timestamp for filename
            
        Returns:
            Path to the JSON report file (or GCS URI if using GCS)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_name_safe = self._sanitize_filename(result.model_name)
        json_path = self._output_dir / f"{model_name_safe}_evaluation_{timestamp}.json"
        
        # Convert to serializable dict
        report_data = self._result_to_dict(result)
        
        # Add metadata
        report_data['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'report_version': '1.0',
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        # Upload to GCS if configured
        json_uri = self._upload_to_gcs(json_path)
        
        logging.info(f"Generated JSON report for {result.model_name}: {json_uri if self._is_gcs else json_path}")
        return json_uri if self._is_gcs else json_path
    
    def _result_to_dict(self, result: ModelEvaluationResult) -> Dict:
        """Convert ModelEvaluationResult to a JSON-serializable dictionary."""
        return {
            'model_name': result.model_name,
            'summary': {
                'total_predictions': result.total_predictions,
                'successful_predictions': result.successful_predictions,
                'failed_predictions': result.failed_predictions,
                'matched_with_ground_truth': result.matched_with_ground_truth,
                'unmatched_count': result.unmatched_count,
                'missing_count': result.missing_count,
            },
            'aggregate_metrics': {
                'endorsed': self._aggregate_to_dict(result.endorsed_aggregate),
                'conflict': self._aggregate_to_dict(result.conflict_aggregate),
                'combined': self._aggregate_to_dict(result.combined_aggregate),
            },
            'per_category_metrics': {
                'endorsed': {
                    cat: self._category_to_dict(metrics)
                    for cat, metrics in result.per_category_endorsed.items()
                },
                'conflict': {
                    cat: self._category_to_dict(metrics)
                    for cat, metrics in result.per_category_conflict.items()
                },
                'combined': {
                    cat: self._category_to_dict(metrics)
                    for cat, metrics in result.per_category_combined.items()
                },
            },
        }
    
    def _aggregate_to_dict(self, aggregate: AggregateResult) -> Dict:
        """Convert AggregateResult to dictionary."""
        return {
            'macro_f1': aggregate.macro_f1,
            'weighted_f1': aggregate.weighted_f1,
            'macro_precision': aggregate.macro_precision,
            'macro_recall': aggregate.macro_recall,
            'categories_evaluated': aggregate.categories_evaluated,
        }
    
    def _category_to_dict(self, category: CategoryResult) -> Dict:
        """Convert CategoryResult to dictionary."""
        return {
            'precision': category.precision,
            'recall': category.recall,
            'f1': category.f1,
            'support': category.support,
            'true_positives': category.true_positives,
            'false_positives': category.false_positives,
            'false_negatives': category.false_negatives,
        }
    
    def generate_comparison_report(
        self,
        results: List[ModelEvaluationResult],
        timestamp: Optional[str] = None,
    ) -> Path:
        """
        Generate a comparison report for multiple models.
        
        Creates a CSV comparing all models side-by-side with:
        - Aggregate metrics for each model
        - Rankings by F1 score
        - Best model per category
        
        Args:
            results: List of model evaluation results to compare
            timestamp: Optional timestamp for filename
            
        Returns:
            Path to the comparison CSV file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        comparison_path = self._output_dir / f"model_comparison_{timestamp}.csv"
        
        with open(comparison_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Section 1: Aggregate Metrics Comparison
            writer.writerow(['=== AGGREGATE METRICS COMPARISON ==='])
            writer.writerow([])
            
            # Header row with model names
            header = ['metric'] + [r.model_name for r in results] + ['best_model']
            writer.writerow(header)
            
            # Endorsed macro F1
            endorsed_macro_f1 = [r.endorsed_aggregate.macro_f1 for r in results]
            best_idx = endorsed_macro_f1.index(max(endorsed_macro_f1)) if endorsed_macro_f1 else -1
            best_model = results[best_idx].model_name if best_idx >= 0 else 'N/A'
            writer.writerow(['endorsed_macro_f1'] + endorsed_macro_f1 + [best_model])
            
            # Endorsed weighted F1
            endorsed_weighted_f1 = [r.endorsed_aggregate.weighted_f1 for r in results]
            best_idx = endorsed_weighted_f1.index(max(endorsed_weighted_f1)) if endorsed_weighted_f1 else -1
            best_model = results[best_idx].model_name if best_idx >= 0 else 'N/A'
            writer.writerow(['endorsed_weighted_f1'] + endorsed_weighted_f1 + [best_model])
            
            # Conflict macro F1
            conflict_macro_f1 = [r.conflict_aggregate.macro_f1 for r in results]
            best_idx = conflict_macro_f1.index(max(conflict_macro_f1)) if conflict_macro_f1 else -1
            best_model = results[best_idx].model_name if best_idx >= 0 else 'N/A'
            writer.writerow(['conflict_macro_f1'] + conflict_macro_f1 + [best_model])
            
            # Conflict weighted F1
            conflict_weighted_f1 = [r.conflict_aggregate.weighted_f1 for r in results]
            best_idx = conflict_weighted_f1.index(max(conflict_weighted_f1)) if conflict_weighted_f1 else -1
            best_model = results[best_idx].model_name if best_idx >= 0 else 'N/A'
            writer.writerow(['conflict_weighted_f1'] + conflict_weighted_f1 + [best_model])
            
            # Combined macro F1
            combined_macro_f1 = [r.combined_aggregate.macro_f1 for r in results]
            best_idx = combined_macro_f1.index(max(combined_macro_f1)) if combined_macro_f1 else -1
            best_model = results[best_idx].model_name if best_idx >= 0 else 'N/A'
            writer.writerow(['combined_macro_f1'] + combined_macro_f1 + [best_model])
            
            # Combined weighted F1
            combined_weighted_f1 = [r.combined_aggregate.weighted_f1 for r in results]
            best_idx = combined_weighted_f1.index(max(combined_weighted_f1)) if combined_weighted_f1 else -1
            best_model = results[best_idx].model_name if best_idx >= 0 else 'N/A'
            writer.writerow(['combined_weighted_f1'] + combined_weighted_f1 + [best_model])
            
            writer.writerow([])
            
            # Section 2: Prediction Statistics
            writer.writerow(['=== PREDICTION STATISTICS ==='])
            writer.writerow([])
            writer.writerow(header[:-1])  # Without 'best_model'
            
            writer.writerow(['total_predictions'] + [r.total_predictions for r in results])
            writer.writerow(['successful_predictions'] + [r.successful_predictions for r in results])
            writer.writerow(['failed_predictions'] + [r.failed_predictions for r in results])
            writer.writerow(['matched_with_ground_truth'] + [r.matched_with_ground_truth for r in results])
            
            writer.writerow([])
            
            # Section 3: Per-Category Endorsed F1 Comparison
            writer.writerow(['=== PER-CATEGORY ENDORSED F1 ==='])
            writer.writerow([])
            writer.writerow(header)
            
            for category in ANNOTATION_CATEGORIES:
                f1_scores = []
                for r in results:
                    cat_result = r.per_category_endorsed.get(category)
                    f1_scores.append(cat_result.f1 if cat_result else 0.0)
                
                best_idx = f1_scores.index(max(f1_scores)) if f1_scores else -1
                best_model = results[best_idx].model_name if best_idx >= 0 else 'N/A'
                writer.writerow([category] + f1_scores + [best_model])
            
            writer.writerow([])
            
            # Section 4: Rankings
            writer.writerow(['=== MODEL RANKINGS (by endorsed weighted F1) ==='])
            writer.writerow([])
            writer.writerow(['rank', 'model', 'endorsed_weighted_f1'])
            
            # Sort by endorsed weighted F1
            ranked = sorted(
                enumerate(results),
                key=lambda x: x[1].endorsed_aggregate.weighted_f1,
                reverse=True
            )
            for rank, (idx, r) in enumerate(ranked, 1):
                writer.writerow([rank, r.model_name, r.endorsed_aggregate.weighted_f1])
        
        # Upload to GCS if configured
        comparison_uri = self._upload_to_gcs(comparison_path)
        
        logging.info(f"Generated comparison report for {len(results)} models")
        return comparison_uri if self._is_gcs else comparison_path
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use in a filename."""
        # Replace problematic characters
        sanitized = name.replace('/', '_').replace('\\', '_')
        sanitized = sanitized.replace(' ', '_').replace(':', '_')
        return sanitized
