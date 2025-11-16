"""
OPTIONAL Mirror Loop Runner

WARNING: This module requires API keys and costs money.
DO NOT run unless explicitly intended.

Estimated cost: ~$0.01-0.05 per run (10 iterations)
"""

from typing import List, Dict, Any, Optional
import warnings

# Flag to prevent accidental execution
RUN_LIVE_DEMO = False


def run_mirror_loop_demo(
    prompt: str = "Explain the concept of recursion in programming.",
    model: str = "gpt-3.5-turbo",
    max_iterations: int = 10,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a minimal Mirror Loop demonstration.

    IMPORTANT: This costs money! Only run if you explicitly want to
    generate new data.

    Args:
        prompt: Initial prompt
        model: Model to use (gpt-3.5-turbo, claude-3-haiku, etc.)
        max_iterations: Maximum iterations
        api_key: API key (or set in environment)

    Returns:
        Dictionary with sequence data
    """
    if not RUN_LIVE_DEMO:
        raise RuntimeError(
            "Live demo is disabled by default. "
            "Set RUN_LIVE_DEMO = True in this file to enable. "
            "This will cost money!"
        )

    warnings.warn(
        f"Running live demo with {model}. "
        f"This will make API calls and cost money (~$0.01-0.05)."
    )

    # Try to import API clients
    try:
        if model.startswith("gpt"):
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        elif model.startswith("claude"):
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported model: {model}")
    except ImportError as e:
        raise ImportError(
            f"API client not available: {e}. "
            "Install with: pip install openai anthropic"
        )

    # Run self-critique loop
    sequence = []
    current_text = None

    for iteration in range(max_iterations):
        if iteration == 0:
            # Initial response
            if model.startswith("gpt"):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                current_text = response.choices[0].message.content
            elif model.startswith("claude"):
                response = client.messages.create(
                    model=model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                current_text = response.content[0].text

        else:
            # Self-critique
            critique_prompt = f"Please review and improve your previous response:\n\n{current_text}"

            if model.startswith("gpt"):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": critique_prompt}],
                    max_tokens=500
                )
                current_text = response.choices[0].message.content
            elif model.startswith("claude"):
                response = client.messages.create(
                    model=model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": critique_prompt}]
                )
                current_text = response.content[0].text

        sequence.append({
            'iteration': iteration,
            'response': current_text,
            'model': model,
        })

        print(f"Iteration {iteration} complete ({len(current_text)} chars)")

    return {
        'sequence_id': 'live_demo',
        'model': model,
        'prompt': prompt,
        'iterations': sequence,
    }


def analyze_live_demo(demo_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze results from a live demo run.

    Args:
        demo_results: Results from run_mirror_loop_demo()

    Returns:
        Analysis results
    """
    from ..metrics import analyze_sequence

    texts = [item['response'] for item in demo_results['iterations']]

    analysis = analyze_sequence(
        texts=texts,
        sequence_id=demo_results['sequence_id'],
        use_embeddings=False
    )

    return analysis
