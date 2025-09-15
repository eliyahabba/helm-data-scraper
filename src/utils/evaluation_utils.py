"""
Evaluation utilities for HELM conversion.

Handles evaluation metrics, prompt format detection, and scoring.
"""

import re
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from config.settings import RUNTIME_METRIC_DATASETS


def get_evaluation_metrics() -> List[Tuple[str, Dict]]:
    """
    Get evaluation metrics in priority order.

    Returns:
        List of (metric_key, method_info) tuples in priority order
    """
    return [
        ("exact_match", {
            "method_name": "label_only_match",
            "description": "Compares only the choice identifier/label to evaluate the response."
        }),
        ("exact_match_indicator", {
            "method_name": "exact_match_indicator",
            "description": "Binary exact match indicator (1 for exact match, 0 otherwise)."
        }),
        ("quasi_exact_match", {
            "method_name": "quasi_label_only_match",
            "description": "Compares only the choice identifier/label to evaluate the response with a tolerance for minor differences."
        }),
        ("final_number_exact_match", {
            "method_name": "final_number_match",
            "description": "Compares the final number of the response with the expected answer."
        }),
        ('math_equiv_chain_of_thought', {
            "method_name": "math_equiv_chain_of_thought",
            "description": "Evaluates the response based on mathematical equivalence in chain of thought reasoning."
        }),
        ('math_equiv', {
            "method_name": "math_equiv",
            "description": "Evaluates mathematical equivalence of the final answer without chain-of-thought."
        }),
        ('f1_score', {
            "method_name": "f1_score",
            "description": "Calculates the F1 score between the predicted and correct answers."
        }),
        ('f1_set_match', {
            "method_name": "f1_set_match",
            "description": "F1 score computed over sets (order-invariant)."
        }),
        ('exact_match_with_references', {
            "method_name": "exact_match_with_references",
            "description": "Compares the predicted answer with the correct answer using references."
        }),
        ('quasi_exact_match_with_references', {
            "method_name": "quasi_exact_match_with_references",
            "description": "Compares the predicted answer with the correct answer using references with a tolerance for minor differences."
        }),
        ('bleu_4', {
            "method_name": "bleu_4",
            "description": "Calculates the BLEU-4 score between the predicted and correct answers."
        }),
        ('rouge_l', {
            "method_name": "rouge_l",
            "description": "Calculates the ROUGE-L score between the predicted and correct answers."
        }),
        ('meteor', {
            "method_name": "meteor_match",
            "description": "Calculates the METEOR score between the predicted and correct answers."
        }),
        ('edit_similarity', {
            "method_name": "edit_similarity",
            "description": "Calculates the edit similarity score between the predicted and correct answers."
        }),
        ('toxic_frac', {
            "method_name": "toxic_frac",
            "description": "Measures the fraction of toxic content in the response (lower is better)."
        }),
        ('pass', {
            "method_name": "pass",
            "description": "Indicates whether the response passed a specific test (1 for pass, 0 for fail)."
        }),
        ('omni_math_accuracy', {
            "method_name": "omni_math_accuracy",
            "description": "Accuracy for Omni-MATH style tasks (value between 0 and 1)."
        }),
        ('wildbench_score_rescaled', {
            "method_name": "wildbench_score_rescaled",
            "description": "Rescaled WildBench score (0 to 1)."
        }),
        ('chain_of_thought_correctness', {
            "method_name": "chain_of_thought_correctness",
            "description": "Correctness of chain-of-thought reasoning (0 to 1)."
        }),
        ('ifeval_strict_accuracy', {
            "method_name": "ifeval_strict_accuracy",
            "description": "IF-Eval strict accuracy (0 to 1)."
        }),
        ('test_avg', {
            "method_name": "test_avg",
            "description": "Average number of tests passed."
        }),
        ('ndcg_10', {
            "method_name": "ndcg_10",
            "description": "Normalized Discounted Cumulative Gain at cutoff 10 (0 to 1)."
        }),
        ('NDCG@10', {
            "method_name": "ndcg_10",
            "description": "Alias for ndcg_10 (Normalized DCG@10)."
        }),
        ('RR@10', {
            "method_name": "rr_10",
            "description": "Reciprocal Rank at cutoff 10 (alias)."
        }),
    ]


def select_evaluation_score(prediction, stats: Dict, dataset_name: Optional[str] = None) -> Tuple[str, float]:
    """
    Select evaluation score using supported metrics in priority order.

    Args:
        stats: Statistics dictionary from prediction
        dataset_name: Optional dataset name for conditional fallbacks

    Returns:
        Tuple of (method_name, score)

    Raises:
        ValueError: If no supported evaluation metric is found
    """
    metric_priority = get_evaluation_metrics()

    for key, method in metric_priority:
        value = stats.get(key)
        if value is not None:
            return method["method_name"], value

    # Conditional fallback: use inference_runtime only for synthetic_efficiency
    if dataset_name in RUNTIME_METRIC_DATASETS:
        runtime_value = stats.get("inference_runtime")
        if runtime_value is not None:
            return "inference_runtime", runtime_value

    # Collect available fields and their values for debugging
    available_fields = {key: value for key, value in stats.items() if value is not None}
    available_info = ", ".join([f"{key}: {value}" for key, value in available_fields.items()])

    raise ValueError(
        f"No supported evaluation metric found in prediction stats. "
        f"Expected one of: exact_match, exact_match_indicator, quasi_exact_match, final_number_exact_match, math_equiv_chain_of_thought, math_equiv, f1_set_match, edit_similarity, toxic_frac, omni_math_accuracy, wildbench_score_rescaled, chain_of_thought_correctness, ifeval_strict_accuracy, test_avg, ndcg_10, NDCG@10, RR@10, inference_runtime (runtime-enabled datasets only). "
        f"Available fields: {available_info}"
        f"Full prediction: {prediction}"
    )


def create_evaluation_section(prediction: Dict, instance: Dict, dataset_name: Optional[str] = None) -> Dict:
    """
    Create evaluation section for evaluation schema.

    Args:
        prediction: Prediction dictionary
        instance: Instance dictionary
        dataset_name: Optional dataset name for conditional fallbacks

    Returns:
        Evaluation section dictionary
    """
    # Find correct answer
    references = instance.get("references", [])
    correct_id = None

    # Identify correct choice letter (A, B, C, D)
    for i, ref in enumerate(references):
        if "correct" in ref.get("tags", []):
            correct_id = chr(65 + i)  # A, B, C, D...
            break

    # Select score using supported metrics
    stats = prediction.get("stats", {}) or {}
    method_name, score = select_evaluation_score(prediction, stats, dataset_name)

    return {
        "ground_truth": correct_id,
        "method_name": method_name,
        "score": score,
    }


def create_output_section(prediction: Dict) -> str:
    """
    Create output section for evaluation schema.

    Args:
        prediction: Prediction dictionary

    Returns:
        Predicted text
    """
    predicted_text = prediction.get("predicted_text")
    if predicted_text:
        predicted_text = predicted_text.strip()
    return predicted_text


def convert_nan_to_null(obj):
    """
    Recursively convert NaN values to None.

    Args:
        obj: Object to process

    Returns:
        Object with NaN values converted to None
    """

    if isinstance(obj, dict):
        return {key: convert_nan_to_null(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_null(item) for item in obj]
    elif isinstance(obj, (float, np.float64, np.float32)) and (math.isnan(obj) or np.isnan(obj)):
        return None
    else:
        return obj
