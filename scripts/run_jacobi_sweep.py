#!/usr/bin/env python3
# scripts/run_jacobi_sweep.py
"""
Run sweep experiments over Jacobi polynomial types and degrees.

Usage:
    python scripts/run_jacobi_sweep.py --sweep types
    python scripts/run_jacobi_sweep.py --sweep degrees
    python scripts/run_jacobi_sweep.py --sweep all
    python scripts/run_jacobi_sweep.py --config configs/jacobi_types/exp_jacobi_legendre.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.runner import run


JACOBI_TYPE_CONFIGS = [
    "configs/jacobi_types/exp_jacobi_legendre.yaml",
    "configs/jacobi_types/exp_jacobi_chebyshev_1st.yaml",
    "configs/jacobi_types/exp_jacobi_chebyshev_2nd.yaml",
    "configs/jacobi_types/exp_jacobi_gegenbauer_1.yaml",
    "configs/jacobi_types/exp_jacobi_gegenbauer_2.yaml",
    "configs/jacobi_types/exp_jacobi_asymmetric_2ab2.yaml",
    "configs/jacobi_types/exp_jacobi_asymmetric_a2b2.yaml",
]

DEGREE_SWEEP_CONFIG = "configs/exp_jacobi_sweep_degrees.yaml"


def load_results(out_dir: Path) -> Dict[str, Any]:
    """Load results.json from experiment output directory."""
    results_path = out_dir / "results.json"
    if not results_path.exists():
        return {}
    return json.loads(results_path.read_text(encoding="utf-8"))


def run_type_sweep() -> List[Dict[str, Any]]:
    """Run sweep over all polynomial types."""
    results = []
    
    for config_path in JACOBI_TYPE_CONFIGS:
        if not Path(config_path).exists():
            print(f"⚠️  Config not found: {config_path}, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Running: {config_path}")
        print(f"{'='*60}")
        
        try:
            out_dir = run(config_path)
            r = load_results(Path(out_dir))
            r["config_path"] = config_path
            r["out_dir"] = str(out_dir)
            results.append(r)
            
            # Print summary
            if "output_sanity" in r:
                print(f"  RMSE: {r['output_sanity'].get('rmse', 'N/A'):.6f}")
                print(f"  Max Abs: {r['output_sanity'].get('max_abs', 'N/A'):.6f}")
        except Exception as e:
            print(f"❌ Error running {config_path}: {e}")
            results.append({
                "config_path": config_path,
                "error": str(e),
            })
    
    return results


def run_degree_sweep() -> List[Dict[str, Any]]:
    """Run sweep over polynomial degrees 2-6."""
    import yaml
    
    results = []
    base_config_path = Path(DEGREE_SWEEP_CONFIG)
    
    if not base_config_path.exists():
        print(f"❌ Degree sweep config not found: {base_config_path}")
        return results
    
    # Load base config
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)
    
    degrees = base_config.get("sweep", {}).get("values", [2, 3, 4, 5, 6])
    
    for degree in degrees:
        print(f"\n{'='*60}")
        print(f"Running degree={degree}")
        print(f"{'='*60}")
        
        # Create temporary config with modified degree
        temp_config = base_config.copy()
        temp_config["float_model"]["arch"]["degree"] = degree
        temp_config["experiment"]["name"] = f"jacobi_degree_{degree}"
        temp_config["logging"]["out_dir"] = f"outputs/exp_runs/jacobi_degree_sweep/degree_{degree}"
        
        # Remove sweep section for single run
        if "sweep" in temp_config:
            del temp_config["sweep"]
        
        temp_config_path = Path(f"/tmp/jacobi_degree_{degree}.yaml")
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f)
        
        try:
            out_dir = run(temp_config_path)
            r = load_results(Path(out_dir))
            r["degree"] = degree
            r["out_dir"] = str(out_dir)
            results.append(r)
            
            # Print summary
            if "output_sanity" in r:
                print(f"  RMSE: {r['output_sanity'].get('rmse', 'N/A'):.6f}")
                print(f"  Max Abs: {r['output_sanity'].get('max_abs', 'N/A'):.6f}")
        except Exception as e:
            print(f"❌ Error running degree={degree}: {e}")
            results.append({
                "degree": degree,
                "error": str(e),
            })
    
    return results


def summarize_results(results: List[Dict[str, Any]], sweep_type: str) -> None:
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {sweep_type}")
    print(f"{'='*80}")
    
    if sweep_type == "types":
        print(f"{'Config':<45} {'RMSE':<12} {'Max Abs':<12} {'Memory (KB)':<12}")
        print("-" * 80)
        for r in results:
            config = Path(r.get("config_path", "")).stem
            if "error" in r:
                print(f"{config:<45} ERROR: {r['error']}")
            else:
                rmse = r.get("output_sanity", {}).get("rmse", float("nan"))
                max_abs = r.get("output_sanity", {}).get("max_abs", float("nan"))
                mem_kb = r.get("memory", {}).get("lut", {}).get("lut_total_bytes", 0) / 1024
                print(f"{config:<45} {rmse:<12.6f} {max_abs:<12.6f} {mem_kb:<12.2f}")
    
    elif sweep_type == "degrees":
        print(f"{'Degree':<10} {'RMSE':<12} {'Max Abs':<12} {'Memory (KB)':<12}")
        print("-" * 50)
        for r in results:
            degree = r.get("degree", "?")
            if "error" in r:
                print(f"{degree:<10} ERROR: {r['error']}")
            else:
                rmse = r.get("output_sanity", {}).get("rmse", float("nan"))
                max_abs = r.get("output_sanity", {}).get("max_abs", float("nan"))
                mem_kb = r.get("memory", {}).get("lut", {}).get("lut_total_bytes", 0) / 1024
                print(f"{degree:<10} {rmse:<12.6f} {max_abs:<12.6f} {mem_kb:<12.2f}")


def save_sweep_results(results: List[Dict[str, Any]], sweep_type: str) -> Path:
    """Save sweep results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"outputs/sweep_results/jacobi_{sweep_type}_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump({
            "sweep_type": sweep_type,
            "timestamp": timestamp,
            "results": results,
        }, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run Jacobi polynomial sweep experiments")
    parser.add_argument(
        "--sweep",
        choices=["types", "degrees", "all"],
        help="Type of sweep to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Run single config file"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to JSON (default: True)"
    )
    
    args = parser.parse_args()
    
    if args.config:
        print(f"Running single config: {args.config}")
        out_dir = run(args.config)
        r = load_results(Path(out_dir))
        print(f"Output: {out_dir}")
        if "output_sanity" in r:
            print(f"RMSE: {r['output_sanity'].get('rmse', 'N/A')}")
            print(f"Max Abs: {r['output_sanity'].get('max_abs', 'N/A')}")
        return
    
    if not args.sweep:
        parser.print_help()
        return
    
    all_results = {}
    
    if args.sweep in ("types", "all"):
        print("\n" + "="*80)
        print("POLYNOMIAL TYPE SWEEP")
        print("="*80)
        results = run_type_sweep()
        summarize_results(results, "types")
        if args.save:
            save_sweep_results(results, "types")
        all_results["types"] = results
    
    if args.sweep in ("degrees", "all"):
        print("\n" + "="*80)
        print("POLYNOMIAL DEGREE SWEEP")
        print("="*80)
        results = run_degree_sweep()
        summarize_results(results, "degrees")
        if args.save:
            save_sweep_results(results, "degrees")
        all_results["degrees"] = results
    
    if args.sweep == "all" and args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = Path(f"outputs/sweep_results/jacobi_all_{timestamp}.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n📁 Combined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
