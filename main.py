import argparse
import sys
from scripts import run_baseline, run_ablation, run_hdcnn

def main():
    parser = argparse.ArgumentParser(description="G14: Multi-Label Wildlife Prediction Pipeline")
    
    # Global arguments
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["baseline", "ablation", "hdcnn", "all"],
                        help="Which part of the pipeline to run.")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs for training (default: 100)")
    
    # Ablation specific arguments
    parser.add_argument("--study", type=str, choices=["arch", "reg", "tta", "all"], 
                        default="all", help="Specific ablation study to run")
    
    # HD-CNN specific arguments
    parser.add_argument("--hierarchy", type=str, choices=["cifar", "learned"], 
                        default="cifar", help="Hierarchy type for HD-CNN")

    args = parser.parse_args()

    try:
        if args.mode == "baseline" or args.mode == "all":
            print("\n=== Starting Baseline Training ===")
            # Assuming run_baseline has a main or execute function
            run_baseline.main(epochs=args.epochs)

        if args.mode == "ablation" or args.mode == "all":
            print(f"\n=== Starting Ablation Studies: {args.study} ===")
            run_ablation.main(study=args.study)

        if args.mode == "hdcnn" or args.mode == "all":
            print(f"\n=== Starting HD-CNN Training (Hierarchy: {args.hierarchy}) ===")
            run_hdcnn.main(hierarchy=args.hierarchy)

    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user. Checkpoints have been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()