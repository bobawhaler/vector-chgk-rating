"""Command line interface for the ChGK Rating System."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import uvicorn

from chgk_rating.analytics import (
    compute_team_synergy_details,
    detect_online_anomalies,
    export_embeddings_to_parquet,
    get_top_players,
    load_model_and_metadata,
)
from chgk_rating.config import (
    DEFAULT_CACHE_DB_PATH,
    DEFAULT_META_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PARQUET_PATH,
    FORMAT_OFFLINE,
    FORMAT_ONLINE,
    FORMAT_SYNC,
    DataIngestConfig,
    ModelConfig,
)
from chgk_rating.data_loader import ChGKDataIngester, generate_synthetic_dataset
from chgk_rating.dataset import ChGKDataset
from chgk_rating.train import evaluate_model, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chgk_cli")


@click.group()
def cli() -> None:
    """ChGK Dual Latent Embedding Rating System CLI."""
    pass


@cli.command("fetch-data")
@click.option("--start-date", default="2023-01-01", help="Start date for tournaments (YYYY-MM-DD)")
@click.option("--max-tournaments", default=30, type=int, help="Max tournaments to fetch")
@click.option(
    "--synthetic",
    is_flag=True,
    default=False,
    help="Generate synthetic benchmark data without external API calls",
)
@click.option(
    "--output-path",
    default=str(DEFAULT_PARQUET_PATH),
    help="Destination path for Parquet dataset",
)
def fetch_data(
    start_date: str, max_tournaments: int, synthetic: bool, output_path: str
) -> None:
    """Fetch tournament results from ChGK rating API or generate synthetic dataset."""
    dest_parquet = Path(output_path)
    if synthetic:
        click.echo("Generating synthetic ChGK dataset...")
        df = generate_synthetic_dataset(output_parquet_path=dest_parquet)
        click.echo(f"Successfully generated {len(df)} observations to {dest_parquet}")
        return

    click.echo(f"Fetching tournaments starting from {start_date} from api.rating.chgk.info...")
    ingest_config = DataIngestConfig(parquet_path=dest_parquet)
    ingester = ChGKDataIngester(ingest_config)

    def progress_callback(msg: str, current: int, total: int) -> None:
        click.echo(f"[{current}/{total}] {msg}")

    try:
        df = asyncio.run(
            ingester.ingest_dataset(
                start_date=start_date,
                max_tournaments=max_tournaments,
                progress_cb=progress_callback,
            )
        )
        if df.empty:
            click.echo("Warning: No records were ingested. You can use --synthetic for offline testing.")
        else:
            click.echo(f"Successfully ingested {len(df)} records into {dest_parquet}")
    except Exception as exc:
        click.echo(f"API ingestion encountered an error: {exc}", err=True)
        click.echo("Falling back to synthetic benchmark dataset generation for local operation...")
        df = generate_synthetic_dataset(output_parquet_path=dest_parquet)
        click.echo(f"Generated {len(df)} synthetic records to {dest_parquet}")


@cli.command("train")
@click.option("--dim-k", default=16, type=int, help="Knowledge latent dimension D")
@click.option("--dim-c", default=8, type=int, help="Communication latent dimension M")
@click.option("--epochs", default=20, type=int, help="Number of training epochs")
@click.option("--batch-size", default=4096, type=int, help="Training batch size")
@click.option("--lr", default=1e-3, type=float, help="Learning rate for AdamW")
@click.option("--weight-decay", default=1e-4, type=float, help="L2 regularization")
@click.option("--data-path", default=str(DEFAULT_PARQUET_PATH), help="Path to Parquet dataset")
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to save best model")
def train(
    dim_k: int,
    dim_c: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    data_path: str,
    model_path: str,
) -> None:
    """Train the Dual Latent Embedding Rating Model."""
    parquet_p = Path(data_path)
    if not parquet_p.exists():
        click.echo(f"Dataset not found at {parquet_p}. Generating synthetic dataset first...")
        generate_synthetic_dataset(output_parquet_path=parquet_p)

    cfg = ModelConfig(
        dim_k=dim_k,
        dim_c=dim_c,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
    )

    click.echo(f"Starting training for {epochs} epochs (D={dim_k}, M={dim_c}, Batch={batch_size})...")
    model, metadata, results = train_model(
        config=cfg,
        parquet_path=parquet_p,
        model_save_path=Path(model_path),
    )

    click.echo("Training Complete!")
    click.echo(f"Best Val LogLoss: {results['best_val_metrics'].get('log_loss', 0.0):.4f}")
    click.echo(f"Test LogLoss:     {results['test_metrics']['log_loss']:.4f}")
    click.echo(f"Test ROC-AUC:     {results['test_metrics']['roc_auc']:.4f}")
    click.echo(f"Test Brier Score: {results['test_metrics']['brier_score']:.4f}")
    if "calibration" in results:
        c = results["calibration"]
        click.echo("--- Learned Data-Driven Temporal Parameters ---")
        click.echo(f"Learned Grace Period:   {c['learned_grace_period_days']:.1f} days")
        click.echo(f"Factual Half-Life:      {c['factual_half_life_days']:.1f} days")
        click.echo(f"Logical Half-Life:      {c['logical_half_life_days']:.1f} days")
        click.echo(f"Synergy Half-Life:      {c['synergy_half_life_days']:.1f} days")
        click.echo(f"Brownian Alpha Gap:     {c['alpha_gap']:.4f}")


@cli.command("evaluate")
@click.option("--data-path", default=str(DEFAULT_PARQUET_PATH), help="Path to Parquet dataset")
@click.option("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to trained model")
@click.option("--meta-path", default=str(DEFAULT_META_PATH), help="Path to dataset metadata")
def evaluate(data_path: str, model_path: str, meta_path: str) -> None:
    """Evaluate trained model on dataset and display metrics."""
    m_path = Path(model_path)
    meta_p = Path(meta_path)
    if not m_path.exists() or not meta_p.exists():
        click.echo("Trained model or metadata not found. Please run 'train' first.", err=True)
        sys.exit(1)

    import torch
    from torch.utils.data import DataLoader
    from chgk_rating.train import split_data_chronologically

    model, metadata = load_model_and_metadata(m_path, meta_p)
    df = pd.read_parquet(data_path)
    _, _, test_df = split_data_chronologically(df)

    test_dataset = ChGKDataset(test_df, metadata)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    metrics = evaluate_model(model, test_loader, torch.device("cpu"))
    click.echo("================ Model Evaluation Metrics ================")
    click.echo(f"Test Samples: {len(test_dataset)}")
    click.echo(f"LogLoss (Cross-Entropy): {metrics['log_loss']:.4f}")
    click.echo(f"ROC-AUC:                 {metrics['roc_auc']:.4f}")
    click.echo(f"Brier Score:             {metrics['brier_score']:.4f}")
    click.echo("==========================================================")


@cli.command("predict-team")
@click.option(
    "--players",
    required=True,
    help="Comma-separated player IDs (e.g., 27403,28751,21026,74001,40393,7008)",
)
@click.option(
    "--format",
    "fmt",
    default=FORMAT_OFFLINE,
    type=click.Choice([FORMAT_OFFLINE, FORMAT_SYNC, FORMAT_ONLINE]),
    help="Tournament format",
)
@click.option("--question-id", default=None, help="Optional question ID to predict")
@click.option("--match-date", default=None, help="Optional match date (YYYY-MM-DD) for time decay")
def predict_team(players: str, fmt: str, question_id: Optional[str], match_date: Optional[str]) -> None:
    """Compute team synergy and predict question outcome for custom roster, with optional time decay."""
    m_path = DEFAULT_MODEL_PATH
    meta_p = DEFAULT_META_PATH
    if not m_path.exists() or not meta_p.exists():
        click.echo("Trained model not found. Please run 'train' first.", err=True)
        sys.exit(1)

    player_list = [p.strip() for p in players.split(",") if p.strip()]
    model, metadata = load_model_and_metadata(m_path, meta_p)

    details = compute_team_synergy_details(
        model, metadata, player_list, format_name=fmt, target_date=match_date
    )
    click.echo("================ Team Synergy Analysis ================")
    click.echo(f"Format:        {details['format']}")
    if match_date:
        click.echo(f"Match Date:    {match_date} (Inactivity Decay Active)")
    click.echo(f"Total Synergy: {details['total_synergy']:.4f}")
    click.echo(f"Players:       {', '.join(details['players'])}")
    if details["best_pair"]:
        bp = details["best_pair"]
        click.echo(
            f"Top Pair:      Player {bp['player_1']} & Player {bp['player_2']} "
            f"(Synergy: {bp['synergy_score']:.4f})"
        )

    if question_id:
        import numpy as np
        import torch
        from datetime import datetime

        q_idx = metadata.question_indexer.get_idx(question_id, default=0)
        p_indices = [metadata.player_indexer.get_idx(p, default=0) for p in player_list[:6]]
        while len(p_indices) < 6:
            p_indices.append(0)

        # Compute delta_t if match_date provided
        player_delta_t: Optional[torch.Tensor] = None
        pairwise_delta_t: Optional[torch.Tensor] = None
        if match_date:
            try:
                m_dt = datetime.strptime(str(match_date)[:10], "%Y-%m-%d")
            except ValueError:
                m_dt = datetime(2026, 9, 1)

            p_deltas = [0.0] * 6
            for i, p_str in enumerate(player_list[:6]):
                if p_str in metadata.player_last_date:
                    last_dt = datetime.strptime(metadata.player_last_date[p_str][:10], "%Y-%m-%d")
                    p_deltas[i] = max(0.0, (m_dt - last_dt).total_seconds() / 86400.0)
            player_delta_t = torch.tensor([p_deltas], dtype=torch.float32)

            pair_mat = np.zeros((1, 6, 6), dtype=np.float32)
            n_p = len(player_list[:6])
            for i in range(n_p):
                for m in range(n_p):
                    if i != m:
                        p1, p2 = sorted([player_list[i], player_list[m]])
                        key = f"{p1}_{p2}"
                        if key in metadata.pair_last_date:
                            l_dt = datetime.strptime(metadata.pair_last_date[key][:10], "%Y-%m-%d")
                            pair_mat[0, i, m] = max(0.0, (m_dt - l_dt).total_seconds() / 86400.0)
                        else:
                            pair_mat[0, i, m] = max(p_deltas[i], p_deltas[m])
            pairwise_delta_t = torch.tensor(pair_mat, dtype=torch.float32)

        p_ten = torch.tensor([p_indices], dtype=torch.long)
        q_ten = torch.tensor([q_idx], dtype=torch.long)
        f_ten = torch.tensor([details["format_id"]], dtype=torch.long)

        with torch.no_grad():
            out = model(
                p_ten,
                q_ten,
                f_ten,
                player_delta_t=player_delta_t,
                pairwise_delta_t=pairwise_delta_t,
            )
            click.echo("---------------- Outcome Prediction ----------------")
            click.echo(f"Question:            {question_id}")
            click.echo(f"Solved Probability:  {out['probs'][0, 0].item():.4f}")
            click.echo(f"Raw Knowledge Prob:  {out['p_gen'][0, 0].item():.4f}")
            click.echo(f"Knowledge Logit:     {out['z_know'][0, 0].item():.4f}")
    click.echo("=======================================================")


@cli.command("player-timeline")
@click.option("--player-id", required=True, help="Player ID to inspect")
@click.option("--start-date", default="2022-01-01", help="Timeline start date (YYYY-MM-DD)")
@click.option("--end-date", default="2026-09-01", help="Timeline end date (YYYY-MM-DD)")
@click.option("--step-days", default=90, type=int, help="Sampling interval in days")
def player_timeline_cmd(player_id: str, start_date: str, end_date: str, step_days: int) -> None:
    """Inspect a player's historical skill trajectory and inactivity decay curve."""
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_META_PATH.exists():
        click.echo("Trained model not found. Run train first.", err=True)
        sys.exit(1)

    from chgk_rating.analytics import get_player_historical_timeline
    model, metadata = load_model_and_metadata(DEFAULT_MODEL_PATH, DEFAULT_META_PATH)
    timeline = get_player_historical_timeline(
        model, metadata, player_id, start_date=start_date, end_date=end_date, step_days=step_days
    )
    if not timeline:
        click.echo(f"Player {player_id} not found in model.")
        return

    click.echo(f"Historical Skill Timeline for Player {player_id}:")
    click.echo(f"{'Date':<12} {'Inactive':<10} {'Eff. ||k||':<12} {'Factual':<10} {'Logical':<10} {'Retention'}")
    click.echo("-" * 65)
    for pt in timeline:
        click.echo(
            f"{pt['date']:<12} {pt['days_inactive']:<10.0f} {pt['effective_knowledge_norm']:<12.4f} "
            f"{pt['factual_norm']:<10.4f} {pt['logical_norm']:<10.4f} {pt['retention_ratio']:<10.1%}"
        )


@cli.command("format-analysis")
def format_analysis_cmd() -> None:
    """Inspect learned relationships, baseline shifts, and geometric similarities among tournament types."""
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_META_PATH.exists():
        click.echo("Trained model not found. Run train first.", err=True)
        sys.exit(1)

    from chgk_rating.analytics import get_format_analysis
    model, metadata = load_model_and_metadata(DEFAULT_MODEL_PATH, DEFAULT_META_PATH)
    analysis = get_format_analysis(model, metadata)

    click.echo("================ Learned Tournament Format Analysis ================")
    click.echo(f"{'Format Name':<12} {'Format ID':<10} {'Baseline Bias':<15} {'Embedding Norm':<15}")
    click.echo("-" * 55)
    for f in analysis["formats"]:
        click.echo(f"{f['format_name']:<12} {f['format_id']:<10} {f['baseline_bias']:<15.4f} {f['embedding_norm']:<15.4f}")

    click.echo("\n--- Pairwise Format Cosine Similarities ---")
    names = analysis["format_names"]
    header = f"{'':<12}" + "".join(f"{n:<12}" for n in names)
    click.echo(header)
    click.echo("-" * len(header))
    for i, row in enumerate(analysis["cosine_similarity_matrix"]):
        row_str = f"{names[i]:<12}" + "".join(f"{val:<12.4f}" for val in row)
        click.echo(row_str)
    click.echo("====================================================================")


@cli.command("top-players")
@click.option("--n", default=15, type=int, help="Number of top players to list")
def top_players_cmd(n: int) -> None:
    """Display top players ranked by knowledge latent vector norm."""
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_META_PATH.exists():
        click.echo("Trained model not found. Run train first.", err=True)
        sys.exit(1)

    model, metadata = load_model_and_metadata(DEFAULT_MODEL_PATH, DEFAULT_META_PATH)
    top = get_top_players(model, metadata, top_n=n)

    click.echo(f"Top {n} Players by Knowledge Vector Norm ||k_i||:")
    click.echo(f"{'Rank':<6} {'Player ID':<15} {'||k_i||':<10}")
    click.echo("-" * 35)
    for p in top:
        click.echo(f"{p['rank']:<6} {p['player_id']:<15} {p['knowledge_norm']:<10.4f}")


@cli.command("detect-anomalies")
@click.option("--data-path", default=str(DEFAULT_PARQUET_PATH), help="Path to Parquet dataset")
def detect_anomalies_cmd(data_path: str) -> None:
    """Detect teams with unnatural online performance spikes (delta_cheat)."""
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_META_PATH.exists():
        click.echo("Trained model not found. Run train first.", err=True)
        sys.exit(1)

    model, metadata = load_model_and_metadata(DEFAULT_MODEL_PATH, DEFAULT_META_PATH)
    df = pd.read_parquet(data_path)
    anomalies = detect_online_anomalies(df, model, metadata, min_games=10)

    click.echo("Top Online Anomaly Scores (delta_cheat):")
    click.echo(f"{'Team ID':<10} {'delta_cheat':<14} {'Online Games':<14} {'Offline Games':<14} {'Suspicious'}")
    click.echo("-" * 65)
    for a in anomalies[:15]:
        susp = "FLAGGED" if a["is_suspicious"] else "Normal"
        click.echo(
            f"{a['team_id']:<10} {a['delta_cheat_score']:<14.4f} "
            f"{a['online_games']:<14} {a['offline_games']:<14} {susp}"
        )


@cli.command("export-embeddings")
@click.option("--output-dir", default="embeddings", help="Directory to save exported files")
def export_embeddings_cmd(output_dir: str) -> None:
    """Export player and question embeddings to Parquet/CSV for UMAP/t-SNE."""
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_META_PATH.exists():
        click.echo("Trained model not found. Run train first.", err=True)
        sys.exit(1)

    model, metadata = load_model_and_metadata(DEFAULT_MODEL_PATH, DEFAULT_META_PATH)
    paths = export_embeddings_to_parquet(model, metadata, Path(output_dir))
    click.echo(f"Successfully exported embeddings to {output_dir}:")
    for k, p in paths.items():
        click.echo(f"  - {k}: {p}")


@cli.command("run-api")
@click.option("--host", default="127.0.0.1", help="Host address to bind")
@click.option("--port", default=8000, type=int, help="Port to listen on")
def run_api_cmd(host: str, port: int) -> None:
    """Launch the FastAPI REST application using Uvicorn."""
    click.echo(f"Starting ChGK Rating REST API at http://{host}:{port} ...")
    uvicorn.run("chgk_rating.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()
