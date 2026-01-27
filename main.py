"""
VAlpha Terminal - Unified Entry Point

This script serves as the main entry point for VAlpha Terminal.
It can run in two modes:
1. Server mode (default): Starts the FastAPI server
2. Analysis mode: Runs pre-market or post-market analysis

Usage:
    # Start the server
    python main.py
    python main.py --host 0.0.0.0 --port 8000

    # Run analysis
    python main.py --mode pre
    python main.py --mode post
"""
import argparse
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_analysis(mode: str):
    """Run pre-market or post-market analysis."""
    from src.analysis.pre_market import PreMarketAnalyst
    from src.analysis.post_market import PostMarketAnalyst
    from src.report_gen import save_report

    print(f"Starting {mode.upper()}-market analysis...")

    if mode == "pre":
        analyst = PreMarketAnalyst()
        report = analyst.run_all()
    elif mode == "post":
        analyst = PostMarketAnalyst()
        report = analyst.run_all()
    else:
        print(f"Unknown mode: {mode}")
        return

    print("\n=== REPORT GENERATED ===\n")
    print(report)

    save_report(report, mode)


def run_server(host: str, port: int, reload: bool = False):
    """Start the FastAPI server."""
    import uvicorn

    print("=" * 50)
    print("VAlpha Terminal")
    print(f"Starting server on http://{host}:{port}")
    print("=" * 50)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload
    )


def main():
    parser = argparse.ArgumentParser(
        description="VAlpha Terminal - Financial Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Start server:     python main.py
  Start with reload: python main.py --reload
  Custom port:      python main.py --port 9000
  Pre-market:       python main.py --mode pre
  Post-market:      python main.py --mode post
        """
    )

    # Server arguments
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    # Analysis arguments
    parser.add_argument(
        "--mode",
        choices=["pre", "post"],
        help="Analysis mode: 'pre' (Pre-market) or 'post' (Post-market)"
    )

    args = parser.parse_args()

    # If mode is specified, run analysis
    if args.mode:
        run_analysis(args.mode)
        return

    # Otherwise, start the server
    run_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
