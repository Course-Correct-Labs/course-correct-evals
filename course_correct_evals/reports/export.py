"""
Report Export Functions

Functions for exporting analysis results and reports.
"""

from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path
import json


def export_csv_results(
    observatory: 'CrossStudyAnalysis',
    output_dir: str = 'results'
) -> Dict[str, str]:
    """
    Export all analysis results to CSV files.

    Args:
        observatory: CrossStudyAnalysis instance with completed analyses
        output_dir: Directory to save CSV files

    Returns:
        Dictionary mapping result type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exported_files = {}

    # Export leaderboard
    try:
        leaderboard = observatory.create_leaderboard()
        leaderboard_path = output_path / 'leaderboard.csv'
        leaderboard.to_csv(leaderboard_path, index=False)
        exported_files['leaderboard'] = str(leaderboard_path)
        print(f"✓ Exported leaderboard to {leaderboard_path}")
    except Exception as e:
        print(f"✗ Error exporting leaderboard: {e}")

    # Export Mirror Loop results
    if observatory.mirror_loop_analysis:
        try:
            seq_analysis = observatory.mirror_loop_analysis['sequence_analysis']
            ml_path = output_path / 'mirror_loop_analysis.csv'
            seq_analysis.to_csv(ml_path, index=False)
            exported_files['mirror_loop'] = str(ml_path)
            print(f"✓ Exported Mirror Loop results to {ml_path}")
        except Exception as e:
            print(f"✗ Error exporting Mirror Loop: {e}")

    # Export Confabulation results
    if observatory.confabulation_analysis:
        try:
            # Export persistence stats as JSON (structured data)
            conf_path = output_path / 'confabulation_analysis.json'
            with open(conf_path, 'w') as f:
                json.dump(
                    observatory.confabulation_analysis['persistence_statistics'],
                    f,
                    indent=2
                )
            exported_files['confabulation'] = str(conf_path)
            print(f"✓ Exported Confabulation results to {conf_path}")
        except Exception as e:
            print(f"✗ Error exporting Confabulation: {e}")

    # Export Violation State results
    if observatory.violation_state_analysis:
        try:
            refusal_stats = observatory.violation_state_analysis['refusal_statistics']
            vs_path = output_path / 'violation_state_analysis.csv'
            refusal_stats.to_csv(vs_path, index=False)
            exported_files['violation_state'] = str(vs_path)
            print(f"✓ Exported Violation State results to {vs_path}")
        except Exception as e:
            print(f"✗ Error exporting Violation State: {e}")

    # Export Echo Chamber results
    if observatory.echo_chamber_analysis:
        try:
            trajectories = observatory.echo_chamber_analysis['trajectories']
            echo_path = output_path / 'echo_chamber_trajectories.csv'
            trajectories.to_csv(echo_path, index=False)
            exported_files['echo_chamber'] = str(echo_path)
            print(f"✓ Exported Echo Chamber results to {echo_path}")
        except Exception as e:
            print(f"✗ Error exporting Echo Chamber: {e}")

    # Export summary
    try:
        summary = observatory.get_summary()
        summary_path = output_path / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        exported_files['summary'] = str(summary_path)
        print(f"✓ Exported summary to {summary_path}")
    except Exception as e:
        print(f"✗ Error exporting summary: {e}")

    return exported_files


def export_pdf_report(
    observatory: 'CrossStudyAnalysis',
    output_path: str = 'ccl_observatory_report.pdf',
    include_figures: bool = True
) -> str:
    """
    Export a comprehensive PDF report.

    Note: This is a simplified version that creates a text report.
    For full PDF generation with figures, additional dependencies would be needed.

    Args:
        observatory: CrossStudyAnalysis instance
        output_path: Path for output PDF
        include_figures: Whether to include figures (requires matplotlib backend)

    Returns:
        Path to generated report
    """
    # For now, create a markdown report that can be converted to PDF
    md_path = output_path.replace('.pdf', '.md')

    with open(md_path, 'w') as f:
        f.write("# CCL Reasoning Stability Observatory Report\n\n")
        f.write("## Executive Summary\n\n")

        summary = observatory.get_summary()

        f.write(f"**Studies Loaded:** {summary['total_studies_loaded']}/4\n\n")

        # Mirror Loop
        if 'mirror_loop' in summary:
            ml = summary['mirror_loop']
            f.write("### Mirror Loop Study\n\n")
            f.write(f"- Total Sequences: {ml['total_sequences']}\n")
            f.write(f"- Collapse Rate: {ml['collapse_rate']:.1%}\n\n")

        # Confabulation
        if 'confabulation' in summary:
            conf = summary['confabulation']
            f.write("### Recursive Confabulation Study\n\n")
            f.write(f"- Total Conversations: {conf['total_conversations']}\n")
            f.write(f"- Persistence Rate: {conf['persistence_rate']:.1%}\n\n")

        # Violation State
        if 'violation_state' in summary:
            vs = summary['violation_state']
            f.write("### Violation State Study\n\n")
            f.write(f"- Total Conversations: {vs['total_conversations']}\n")
            f.write(f"- Contamination Rate: {vs['contamination_rate']:.1%}\n\n")

        # Echo Chamber
        if 'echo_chamber' in summary:
            echo = summary['echo_chamber']
            f.write("### Echo Chamber Study\n\n")
            f.write(f"- Total Simulations: {echo['total_simulations']}\n\n")

        # Leaderboard
        f.write("## Model Leaderboard\n\n")
        try:
            leaderboard = observatory.create_leaderboard()
            f.write(leaderboard.to_markdown(index=False))
            f.write("\n\n")
        except Exception as e:
            f.write(f"Error generating leaderboard: {e}\n\n")

        f.write("---\n\n")
        f.write("*Generated by CCL Reasoning Stability Observatory*\n")

    print(f"✓ Markdown report exported to {md_path}")
    print(f"  (Convert to PDF using pandoc or similar tool)")

    return md_path
