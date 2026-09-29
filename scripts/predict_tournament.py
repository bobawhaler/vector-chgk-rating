"""Run tournament forecast for an arbitrary tournament ID."""

import argparse
import sys
from pathlib import Path

from chgk_rating.tournament_predictor import TournamentPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict tournament standings, expected scores, and win probabilities for any tournament ID."
    )
    parser.add_argument("tournament_id", type=int, help="Official tournament ID (e.g. 14149, 10000, 10725)")
    parser.add_argument("--simulations", type=int, default=10000, help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--tactical", action="store_true", default=True, help="Enable tactical tour-by-tour roster optimization")
    parser.add_argument("--no-tactical", action="store_false", dest="tactical")
    parser.add_argument("--refresh", action="store_true", default=False, help="Force refresh tournament data from official API")
    parser.add_argument("--dl-override", type=float, default=None, help="Override package TrueDL for sensitivity analysis")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for JSON and HTML reports")
    parser.add_argument("--no-html", action="store_true", default=False, help="Do not generate HTML report")

    args = parser.parse_args()

    predictor = TournamentPredictor()
    print(f"Predicting outcomes for tournament ID {args.tournament_id}...")

    try:
        res = predictor.predict_tournament(
            tournament_id=args.tournament_id,
            num_simulations=args.simulations,
            tactical=args.tactical,
            force_refresh=args.refresh,
            tournament_dl_override=args.dl_override,
        )
    except Exception as exc:
        print(f"Error predicting tournament {args.tournament_id}: {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    json_path = predictor.export_json(res, out_dir / f"forecast_{args.tournament_id}.json")
    print(f"Saved forecast JSON to {json_path}")

    if not args.no_html:
        html_path = predictor.export_html(res, out_dir / f"forecast_{args.tournament_id}.html")
        print(f"Saved interactive HTML report to {html_path}")

    # Output formatted terminal table
    print("\n" + predictor.format_terminal_table(res) + "\n")


if __name__ == "__main__":
    main()
