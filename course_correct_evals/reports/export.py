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

    # Export Mirror Loop results.
    # Provenance-separated, per the same pattern used for Confabulation and
    # Violation State: the PRIMARY canonical plateau result (tau=0.05,
    # per-sequence-then-aggregated) is never blended in the same file as
    # the explicitly secondary tau=0.02 sensitivity view or the distinct
    # grounding-rebound finding.
    if observatory.mirror_loop_analysis:
        try:
            plateau = observatory.mirror_loop_analysis['plateau']

            seq_path = output_path / 'mirror_loop_plateau_sequences.csv'
            plateau['sequence_results'].to_csv(seq_path, index=False)
            exported_files['mirror_loop_plateau_sequences'] = str(seq_path)
            print(f"✓ Exported Mirror Loop per-sequence plateau results (tau=0.05, primary) to {seq_path}")

            group_rows = []
            for group_key, stats in plateau['group_summary'].items():
                row = {gc: gv for gc, gv in zip(plateau['group_cols'], group_key)}
                row.update(stats)
                group_rows.append(row)
            group_path = output_path / 'mirror_loop_plateau_group_summary.csv'
            pd.DataFrame(group_rows).to_csv(group_path, index=False)
            exported_files['mirror_loop_plateau_group_summary'] = str(group_path)
            print(f"✓ Exported Mirror Loop plateau group summary (tau=0.05, primary) to {group_path}")

            sens = observatory.mirror_loop_analysis['plateau_sensitivity_tau_0_02']
            sens_rows = []
            for group_key, stats in sens['group_summary'].items():
                row = {gc: gv for gc, gv in zip(sens['group_cols'], group_key)}
                row.update(stats)
                sens_rows.append(row)
            sens_path = output_path / 'mirror_loop_plateau_sensitivity_tau_0_02.csv'
            pd.DataFrame(sens_rows).to_csv(sens_path, index=False)
            exported_files['mirror_loop_plateau_sensitivity_tau_0_02'] = str(sens_path)
            print(f"✓ Exported Mirror Loop plateau sensitivity view (tau=0.02, secondary, does not feed leaderboard) to {sens_path}")

            rebound = observatory.mirror_loop_analysis['grounding_rebound']
            if rebound is not None:
                rebound_path = output_path / 'mirror_loop_grounding_rebound.csv'
                pd.DataFrame([rebound]).to_csv(rebound_path, index=False)
                exported_files['mirror_loop_grounding_rebound'] = str(rebound_path)
                print(f"✓ Exported Mirror Loop grounding-rebound finding (distinct from plateau) to {rebound_path}")
        except Exception as e:
            print(f"✗ Error exporting Mirror Loop: {e}")

    # Export Confabulation results.
    # Two separate files, per provenance separation: released measurements
    # are never interleaved with Observatory-derived pooled values in the
    # same table.
    #   File A: the released model x arm table, one row per (model, arm),
    #           every released column preserved.
    #   File B: the manuscript-defined pooled intervention comparison
    #           (baseline/fact_table/belief_audit only, N-weighted;
    #           grounding_pilot deliberately excluded) with explicit
    #           provenance labeling.
    if observatory.confabulation_analysis and observatory.confabulation_analysis.get('data_type') == 'model_arm_table':
        try:
            model_arm_path = output_path / 'confabulation_model_arm.csv'
            observatory.confabulation_analysis['model_arm_table'].to_csv(model_arm_path, index=False)
            exported_files['confabulation_model_arm'] = str(model_arm_path)
            print(f"✓ Exported Confabulation model x arm table to {model_arm_path}")

            pooled = observatory.confabulation_analysis['pooled_intervention_comparison']
            pooled_rows = [
                {
                    'arm': arm,
                    'n': stats['n'],
                    'persist_count': round(stats['persist_rate'] * stats['n']),
                    'persist_rate': stats['persist_rate'],
                    'provenance': 'observatory_pooled_manuscript_defined',
                    'weighting': 'N-weighted across models',
                }
                for arm, stats in pooled.items()
            ]
            pooled_path = output_path / 'confabulation_pooled_intervention_comparison.csv'
            pd.DataFrame(pooled_rows).to_csv(pooled_path, index=False)
            exported_files['confabulation_pooled_comparison'] = str(pooled_path)
            print(f"✓ Exported Confabulation pooled intervention comparison to {pooled_path}")
        except Exception as e:
            print(f"✗ Error exporting Confabulation: {e}")
    elif observatory.confabulation_analysis:
        # Legacy per-conversation mode; unreachable via the normal
        # importer path (see CrossStudyAnalysis.analyze_confabulation()).
        try:
            conf_path = output_path / 'confabulation_analysis.json'
            with open(conf_path, 'w') as f:
                json.dump(
                    observatory.confabulation_analysis.get('persistence_statistics', {}),
                    f,
                    indent=2
                )
            exported_files['confabulation'] = str(conf_path)
            print(f"✓ Exported Confabulation results to {conf_path}")
        except Exception as e:
            print(f"✗ Error exporting Confabulation: {e}")

    # Export Violation State results
    # Flattened to one row per condition, preserving BOTH provenance
    # layers as distinct columns: the raw structured outcomes (as
    # released, terminal rate_limit kept distinct) and the
    # published/historical aggregate (rate-limit-as-refusal convention).
    if observatory.violation_state_analysis:
        try:
            structured = observatory.violation_state_analysis['structured']
            raw = structured['raw_structured_outcomes']
            published = structured['published_aggregate']

            vs_rows = []
            for cond in raw:
                counts = raw[cond]['counts']
                pub = published[cond]
                vs_rows.append({
                    'condition': cond,
                    'n': raw[cond]['n'],
                    'policy_refusal': counts.get('policy_refusal', 0),
                    'capability_refusal': counts.get('capability_refusal', 0),
                    'image_success': counts.get('image_success', 0),
                    'terminal_rate_limit': counts.get('rate_limit', 0),
                    'published_refused': pub['refused'],
                    'published_refusal_rate': pub['refusal_rate'],
                })

            vs_df = pd.DataFrame(vs_rows)
            vs_path = output_path / 'violation_state_analysis.csv'
            vs_df.to_csv(vs_path, index=False)
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

        f.write(f"**Studies Loaded:** {summary['total_studies_loaded']}/3\n\n")

        # Mirror Loop
        if 'mirror_loop' in summary:
            ml = summary['mirror_loop']
            f.write("### Mirror Loop Study\n\n")
            f.write(f"- Total Sequences: {ml['total_sequences']}\n")
            f.write(f"- Mean ΔI (released edit_change, overall): {ml['mean_delta_i_overall']:.3f}\n\n")

            f.write("**PLATEAU RATE** (manuscript-defined rolling-3-step statistic, "
                    "tau=0.05 PRIMARY, per-sequence detection then aggregated -- "
                    "never a pooled-trajectory crossing):\n\n")
            for label, stats in ml['plateau_group_summary'].items():
                iqr_txt = ""
                if stats['median_plateau_iteration'] is not None:
                    iqr_txt = (f", median iter {stats['median_plateau_iteration']:.0f} "
                               f"(IQR {stats['plateau_iteration_iqr'][0]:.0f}-"
                               f"{stats['plateau_iteration_iqr'][1]:.0f})")
                f.write(f"- {label}: {stats['n_plateaued']}/{stats['n_sequences']} plateaued "
                        f"({stats['plateau_rate']:.1%}){iqr_txt}\n")
            f.write("\n")

            f.write("**SENSITIVITY VIEW** (tau=0.02, SECONDARY -- does not feed the leaderboard "
                    "and does not alter the tau=0.05 primary result above):\n\n")
            for label, stats in ml['plateau_sensitivity_tau_0_02_group_summary'].items():
                f.write(f"- {label}: {stats['n_plateaued']}/{stats['n_sequences']} plateaued "
                        f"({stats['plateau_rate']:.1%})\n")
            f.write("\n")

            rebound = ml['grounding_rebound']
            if rebound is not None:
                f.write("**GROUNDING REBOUND** (manuscript-defined pooled ΔI comparison, "
                        "grounded condition only; a DISTINCT finding from plateau, not derived "
                        "from or combined with it):\n\n")
                f.write(f"- Iteration {rebound['iteration_from']} → {rebound['iteration_to']}: "
                        f"{rebound['delta_i_from']:.6f} → {rebound['delta_i_to']:.6f} "
                        f"({rebound['pct_increase']:+.1f}%)\n\n")

        # Confabulation
        if 'confabulation' in summary:
            conf = summary['confabulation']
            f.write("### Recursive Confabulation Study\n\n")

            if conf.get('data_type') == 'model_arm_table':
                f.write(f"- Models: {', '.join(conf['models'])}\n")
                f.write(f"- Intervention arms: {', '.join(conf['arms'])}\n")
                f.write(f"- Total conversations: {conf.get('total_conversations', 'N/A')}\n\n")

                f.write("**A. RELEASED MODEL x ARM RESULTS** "
                        "(source measurements, no arm collapse):\n\n")
                f.write("See the exported `confabulation_model_arm.csv` for the full "
                        "released table (one row per (model, arm)).\n\n")

                f.write("**B. MANUSCRIPT-DEFINED POOLED INTERVENTION COMPARISON** "
                        "(N-weighted across models, persist_rate):\n\n")
                for arm, stats in conf['pooled_intervention_comparison'].items():
                    f.write(f"- {arm}: {stats['persist_rate']:.2%} (N={stats['n']})\n")
                f.write("\n")

                f.write("**C. GROUNDING-CONFABULATION HETEROGENEITY** "
                        "(model-specific, confab_rate; NOT part of the pooled three-arm "
                        "persistence comparison above -- a different outcome variable):\n\n")
                for model, stats in conf['grounding_confabulation_heterogeneity'].items():
                    f.write(f"- {model}: {stats['confab_rate']:.2%} (N={stats['n']})\n")
                f.write("\nSource study finding: \"Grounding reduced confabulation for GPT-4o "
                        "mini only\" (README.md / RC_publication_pack.md). Per the manuscript "
                        "(Section 4.3, not present in the audited repository's artifacts), "
                        "GPT-4o Mini alone was statistically significant (p=0.033); that figure "
                        "is manuscript-sourced and not re-derived here. Grounding's effect on "
                        "persistence specifically is a separate released measurement (see the "
                        "model x arm export or the confab_persist_rate_grounding_pilot "
                        "leaderboard column) and is not this confabulation finding.\n\n")
            else:
                # Legacy per-conversation mode; unreachable via the normal
                # importer path.
                total_convs = conf.get('total_conversations', 'N/A')
                f.write(f"- Total Conversations: {total_convs}\n")
                persistence = conf.get('persistence_rate')
                if persistence is not None:
                    f.write(f"- Persistence Rate: {persistence:.1%}\n")
                f.write("\n")

        # Violation State
        if 'violation_state' in summary:
            vs = summary['violation_state']
            f.write("### Violation State Study\n\n")
            f.write(f"- Total Conversations: {vs['total_conversations']}\n\n")

            f.write("**RAW STRUCTURED OUTCOMES** (released data, as observed):\n\n")
            for cond, data in vs['raw_structured_outcomes'].items():
                f.write(f"- {cond} (N={data['n']}): {data['counts']}\n")

            f.write("\n**PUBLISHED/HISTORICAL AGGREGATE** "
                    "(historical rate-limit-as-refusal convention):\n\n")
            for cond, data in vs['published_aggregate'].items():
                f.write(f"- {cond}: {data['refused']}/{data['n']} refused ({data['refusal_rate']:.2%})\n")

            f.write("\nThe raw released data contain one terminal, never-retried rate_limit "
                    "(contaminated condition). The historical final-analysis convention used by "
                    "the study counted that terminal rate_limit in the refusal/failure aggregate, "
                    "producing the published figure above. It is not presented here as an observed "
                    "policy refusal.\n\n")

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
