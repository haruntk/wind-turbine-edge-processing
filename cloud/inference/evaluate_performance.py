#!/usr/bin/env python3
"""Model Performance Evaluation Script.

Connects to TimescaleDB, joins anomaly results with feature vectors,
derives true anomaly labels (scenario_label != 'healthy'),
and prints standard classification metrics (precision, recall, F1, accuracy)
along with detailed detection rates for each fault scenario.

Usage:
    python cloud/inference/evaluate_performance.py
"""

from __future__ import annotations

import os
import sys
import logging
import psycopg2

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluator")


def print_table(headers, rows):
    """Prints a beautiful table in ASCII grid format without external dependencies."""
    if not headers and not rows:
        return
        
    # Find max length of each column
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            val_str = str(val)
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val_str))
            else:
                col_widths.append(len(val_str))
                
    # Build borders
    border = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_sep = "+" + "+".join(["=" * (w + 2) for w in col_widths]) + "+"
    
    # Print headers
    print(border)
    header_cols = [f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)]
    print("|" + "|".join(header_cols) + "|")
    print(header_sep)
    
    # Print rows
    for row in rows:
        row_cols = []
        for i, val in enumerate(row):
            w = col_widths[i] if i < len(col_widths) else 10
            row_cols.append(f" {str(val):<{w}} ")
        print("|" + "|".join(row_cols) + "|")
        print(border)



def connect_db():
    host = os.environ.get("DB_HOST", "127.0.0.1")
    port = int(os.environ.get("DB_PORT", 5433))
    name = os.environ.get("DB_NAME", "windguard")
    user = os.environ.get("DB_USER", "windguard_user")
    pwd = os.environ.get("DB_PASSWORD", "ceren123")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=name,
            user=user,
            password=pwd,
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to TimescaleDB at {host}:{port}/{name}: {e}")
        logger.error("Make sure your gcloud IAP tunnel is running (localhost:5433 -> windguard-db:5432).")
        sys.exit(1)


def main():
    conn = connect_db()
    
    logger.info("Fetching joined anomaly results and feature vectors from database...")
    
    query = """
        SELECT 
            ar.time,
            fv.scenario_label,
            ar.reconstruction_error,
            ar.threshold,
            ar.is_anomaly
        FROM anomaly_results ar
        JOIN feature_vectors fv ON ar.time = fv.time
        ORDER BY ar.time ASC
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to execute evaluation query: {e}")
        conn.close()
        sys.exit(1)
        
    conn.close()
    
    if not rows:
        print("\n[!] No matching records found. Make sure both 'feature_vectors' and 'anomaly_results' have data.")
        print("Please ensure your simulator is running and publishing vectors, and the inference service is processing them.")
        return

    total = len(rows)
    print(f"\n==============================================================")
    print(f"Model Performance Evaluation Report (Total Evaluated Windows: {total})")
    print(f"==============================================================\n")

    # Metrics counters
    tp = 0  # True Positive: Faulty predicted Anomaly
    fp = 0  # False Positive: Healthy predicted Anomaly
    tn = 0  # True Negative: Healthy predicted Normal
    fn = 0  # False Negative: Faulty predicted Normal

    # Scenario stats
    # Structure: {scenario: {"total": 0, "anomalies": 0, "normals": 0}}
    scenario_stats = {}

    for row in rows:
        time_stamp, scenario_label, recon_err, threshold, is_anomaly = row
        
        # Determine Ground Truth: Anything other than "healthy" is anomalous/faulty
        is_actually_faulty = scenario_label.lower() != "healthy"
        
        # Update confusion matrix
        if is_actually_faulty:
            if is_anomaly:
                tp += 1
            else:
                fn += 1
        else:
            if is_anomaly:
                fp += 1
            else:
                tn += 1

        # Update scenario statistics
        if scenario_label not in scenario_stats:
            scenario_stats[scenario_label] = {"total": 0, "anomalies": 0, "normals": 0}
            
        scenario_stats[scenario_label]["total"] += 1
        if is_anomaly:
            scenario_stats[scenario_label]["anomalies"] += 1
        else:
            scenario_stats[scenario_label]["normals"] += 1

    # Calculations
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Summary table
    summary_data = [
        ["Accuracy", f"{accuracy:.4f} ({accuracy * 100:.2f}%)"],
        ["Precision (PPV)", f"{precision:.4f} ({precision * 100:.2f}%)"],
        ["Recall (Sensitivity)", f"{recall:.4f} ({recall * 100:.2f}%)"],
        ["F1-Score", f"{f1:.4f}"],
        ["False Positive Rate (FPR)", f"{fpr:.4f} ({fpr * 100:.2f}%)"],
    ]
    print("Core Classification Metrics:")
    print_table(["Metric", "Value"], summary_data)
    print()

    # Confusion matrix table
    cm_data = [
        ["Actual Faulty", f"TP: {tp}", f"FN: {fn}", f"Total: {tp + fn}"],
        ["Actual Healthy", f"FP: {fp}", f"TN: {tn}", f"Total: {fp + tn}"],
    ]
    print("Confusion Matrix:")
    print_table(["", "Predicted Anomaly", "Predicted Normal", "Total"], cm_data)
    print()

    # Detailed scenario-by-scenario breakdown
    breakdown_data = []
    for sc, stats in sorted(scenario_stats.items()):
        tot = stats["total"]
        anom = stats["anomalies"]
        norm = stats["normals"]
        rate = anom / tot if tot > 0 else 0
        
        # Ground truth status string
        gt_type = "Healthy" if sc.lower() == "healthy" else "Faulty"
        
        # Detection rate or false alarm rate label
        rate_label = "False Alarm Rate" if sc.lower() == "healthy" else "Detection Rate"
        
        breakdown_data.append([
            sc,
            gt_type,
            tot,
            anom,
            norm,
            f"{rate * 100:.1f}% ({rate_label})"
        ])

    print("Detailed Scenario Breakdown:")
    print_table(
        ["Scenario Label", "Type", "Total Samples", "Pred Anomalies", "Pred Normals", "Rate"],
        breakdown_data
    )
    print()


if __name__ == "__main__":
    main()
