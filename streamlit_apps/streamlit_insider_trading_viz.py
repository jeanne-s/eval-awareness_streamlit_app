import html
import json
import os
from io import BytesIO
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Projection on the eval-awareness probe scores",
    page_icon="📊",
    layout="wide",
)

# Initialize session state for caching
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "jsonl_data" not in st.session_state:
    st.session_state.jsonl_data = None

# Placeholder: Fill this with your actual available layers and datasets
AVAILABLE_PROBES = {
    "25": ["insider_trading.jsonl", "triggers_qwq.jsonl"],
    "30": ["insider_trading.jsonl", "triggers_qwq.jsonl"],
    "35": ["insider_trading.jsonl"],
    "40": ["insider_trading.jsonl"],
    "45": ["insider_trading.jsonl"],
    "50": ["insider_trading.jsonl"],
    "55": ["insider_trading.jsonl"],
    "58": ["insider_trading.jsonl", "triggers_qwq.jsonl"],
    "60": ["insider_trading.jsonl"],
    # Add more layers and datasets as needed
}


@st.cache_data
def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_token_visualization(
    tokens: List[str],
    scores: List[float],
    title: str = "Token Scores",
    color_scheme: str = "RdYlBu_r",
    score_type: str = "projections",
) -> go.Figure:
    """Create a heatmap visualization of token scores."""

    # Use fixed normalization for probabilities
    if score_type == "probability_class_0":
        norm = plt.Normalize(0.0, 1.0)
    else:
        norm = plt.Normalize(min(scores), max(scores))
    cmap = plt.cm.get_cmap(color_scheme)
    colors = [mcolors.rgb2hex(cmap(norm(score))) for score in scores]

    # For very long token sequences, use a more efficient approach
    if len(tokens) > 200:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores))),
                y=[0] * len(scores),
                mode="markers",
                marker=dict(size=20, color=colors, line=dict(width=1, color="black")),
                text=[
                    f"Token: {token}<br>Score: {score:.4f}"
                    for token, score in zip(tokens, scores)
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Token Scores",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Token Position",
            yaxis_title="",
            height=200,
            showlegend=False,
            yaxis=dict(showticklabels=False, range=[-0.5, 0.5]),
        )
        return fig
    # For shorter sequences, use the original annotation approach
    fig = go.Figure()
    for i, (token, score, color) in enumerate(zip(tokens, scores, colors)):
        display_token = token.replace("\n", "<br>").replace(" ", "&nbsp;")
        if display_token == "":
            display_token = "&nbsp;"
        fig.add_annotation(
            x=i,
            y=0,
            text=display_token,
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor=color,
            bordercolor="black",
            borderwidth=1,
            align="center",
            valign="middle",
            width=40,
            height=30,
        )
    fig.update_layout(
        title=title,
        xaxis=dict(
            showgrid=False, showticklabels=False, range=[-0.5, len(tokens) - 0.5]
        ),
        yaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 0.5]),
        plot_bgcolor="white",
        height=150,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_score_visualization(
    tokens: List[str],
    scores: List[float],
    title: str = "Token Scores",
    color_scheme: str = "RdYlBu_r",
) -> go.Figure:
    """Create a bar chart visualization of token scores."""

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(len(scores))),
            y=scores,
            marker=dict(color=scores, colorscale=color_scheme),
            name="Token Scores",
            hovertemplate="Token: %{text}<br>Position: %{x}<br>Score: %{y:.4f}<extra></extra>",
            text=tokens,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Token Position",
        yaxis_title="Score",
        height=400,
        showlegend=False,
    )

    return fig


def create_score_distribution_plot(
    scores: List[float], title: str = "Score Distribution"
) -> go.Figure:
    """Create a histogram of score distribution."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=30,
            name="Score Distribution",
            marker_color="lightblue",
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=title, xaxis_title="Score", yaxis_title="Frequency", showlegend=False
    )

    return fig


def create_text_with_highlighted_tokens(
    tokens: List[str],
    scores: List[float],
    color_scheme: str = "RdYlBu_r",
    score_type: str = "projections",
) -> str:
    """Create HTML text with all tokens colored according to their score (heatmap style), preserving original text flow."""
    if len(tokens) != len(scores):
        return "Error: Token count mismatch"

    # Use fixed normalization for probabilities, min/max for projections
    if score_type == "probability_class_0":
        norm = plt.Normalize(0.0, 1.0)
    else:
        norm = plt.Normalize(min(scores), max(scores))
    cmap = plt.cm.get_cmap(color_scheme)
    colors = [mcolors.rgb2hex(cmap(norm(score))) for score in scores]

    html_parts = []
    for token, color, score in zip(tokens, colors, scores):
        # Escape HTML for safety
        safe_token = html.escape(token)
        # If the token is just whitespace, don't wrap it in a span
        if safe_token.strip() == "":
            html_parts.append(safe_token.replace("\n", "<br>"))
        else:
            html_parts.append(
                f'<span style="background-color: {color}; font-weight: bold;" title="Score: {score:.4f}">{safe_token}</span>'
            )
    return "".join(html_parts)


def create_colorbar(
    min_val: float, max_val: float, color_scheme: str = "RdYlBu_r", title: str = "Score"
):
    """Create a large vertical colorbar using matplotlib and display as an image."""
    fig, ax = plt.subplots(figsize=(0.5, 2))  # width, height in inches
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    mpl.colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(color_scheme), norm=norm, orientation="vertical"
    )
    ax.set_title(title, fontsize=16)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


def main():
    st.title("What aspects of the insider trading prompts induce eval-awareness?")

    # Sidebar controls
    st.sidebar.header("Controls")

    # Layer selection
    layer_options = sorted(AVAILABLE_PROBES.keys(), key=lambda x: int(x))
    selected_layer = st.sidebar.selectbox(
        "Select Layer", options=layer_options, format_func=str
    )

    # Dataset selection for the selected layer
    dataset_options = AVAILABLE_PROBES[selected_layer]
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=dataset_options,
        format_func=lambda x: x.replace(".jsonl", "").replace("_", " ").title(),
    )

    # Build the file path
    selected_file = (
        f"output_data/probe_projections/QwQ-32B/{selected_layer}/{selected_dataset}"
    )

    # Check if files exist
    if not os.path.exists(selected_file):
        st.error(f"JSONL file not found: {selected_file}")
        return

    # Load data if not already loaded or if file changed
    if (
        not st.session_state.data_loaded
        or "current_file" not in st.session_state
        or st.session_state.current_file != selected_file
    ):
        with st.spinner("Loading data..."):
            st.session_state.jsonl_data = load_jsonl_data(selected_file)
            st.session_state.data_loaded = True
            st.session_state.current_file = selected_file
            st.success(f"Loaded {len(st.session_state.jsonl_data)} samples")

    # Sample selection
    sample_idx = st.sidebar.slider(
        "Select Sample",
        min_value=0,
        max_value=len(st.session_state.jsonl_data) - 1,
        value=0,
        step=1,
        format="%d",
    )
    st.sidebar.write(
        f"Sample {sample_idx} (Text Index: {st.session_state.jsonl_data[sample_idx]['text_index']})"
    )

    # Score type selection
    score_type_options = {
        "P(Deployment)": "probability_class_0",
        "Projection Scores": "projections",
    }

    selected_score_type_display = st.sidebar.selectbox(
        "Score Type",
        options=list(score_type_options.keys()),
        help="Choose which scores to visualize",
    )

    score_type = score_type_options[selected_score_type_display]

    # Color scheme selection
    color_scheme = st.sidebar.selectbox(
        "Color Scheme", options=["cool"], help="Choose color scheme for visualization"
    )

    # Visualization type selection
    viz_type = "Token Heatmap"

    # Token limit for long sequences
    if len(st.session_state.jsonl_data[sample_idx].get("tokens", [])) > 200:
        max_tokens = st.sidebar.slider(
            "Max Tokens to Display",
            min_value=50,
            max_value=min(
                1500, len(st.session_state.jsonl_data[sample_idx].get("tokens", []))
            ),
            value=min(
                500, len(st.session_state.jsonl_data[sample_idx].get("tokens", []))
            ),
            help="Limit tokens for better performance",
        )
    else:
        max_tokens = len(st.session_state.jsonl_data[sample_idx].get("tokens", []))

    # Get selected sample
    sample = st.session_state.jsonl_data[sample_idx]
    text_index = sample["text_index"]

    # Check if tokens are available
    if "tokens" not in sample:
        st.error(
            "No tokens found in the JSONL file. Please run the probe projection pipeline."
        )
        return

    # Get scores and tokens
    if score_type == "projections":
        scores = sample["projections"]
        title = "Projection Scores"
    else:
        scores = sample["probability_class_0"]
        title = "P(Deployment) = 1 - P(Eval)"

    tokens = sample["tokens"]

    # Apply token limit for performance
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        scores = scores[:max_tokens]

    # Ensure scores match token count
    if len(scores) != len(tokens):
        st.warning(
            f"Score count ({len(scores)}) doesn't match token count ({len(tokens)})"
        )
        # Truncate to match the shorter length
        min_length = min(len(scores), len(tokens))
        scores = scores[:min_length]
        tokens = tokens[:min_length]

    # Show all metadata fields for the selected sample
    metadata = {
        k: v
        for k, v in sample.items()
        if k not in ["projections", "probability_class_0", "tokens", "text_index"]
    }
    if metadata:
        st.sidebar.markdown("**Sample Metadata:**")
        for k, v in metadata.items():
            st.sidebar.write(f"**{k}:** {v}")

    # Main content
    col1, col2, col3 = st.columns([2, 1, 0.3])

    with col1:
        if len(tokens) > 500:
            st.warning(
                f"⚠️ Large token sequence ({len(tokens)} tokens). Visualization may take a moment to load."
            )

        # Only show the token heatmap visualization
        fig = create_token_visualization(
            tokens, scores, title, color_scheme, score_type
        )
        st.plotly_chart(fig, use_container_width=True)

        highlighted_text = create_text_with_highlighted_tokens(
            tokens, scores, color_scheme=color_scheme, score_type=score_type
        )
        st.markdown(highlighted_text, unsafe_allow_html=True)

        # Statistics and top tokens side by side
        col_stats, col_tokens = st.columns([2, 1])

        with col_stats:
            st.subheader("Statistics")

            # Calculate statistics
            scores_array = np.array(scores)
            stats = {
                "Mean": np.mean(scores_array),
                "Std": np.std(scores_array),
                "Min": np.min(scores_array),
                "Max": np.max(scores_array),
                "Median": np.median(scores_array),
            }

            for stat_name, stat_value in stats.items():
                st.metric(stat_name, f"{stat_value:.4f}")

            # Score distribution
            fig_dist = create_score_distribution_plot(scores, f"{title} Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)

            # Token summary
            st.write(f"**Total tokens:** {len(scores)}")

        with col_tokens:
            # Top and bottom scoring tokens side by side
            col_top, col_bottom = st.columns(2)

            with col_top:
                st.write("**Top 20 DEPLOYMENT tokens:**")
                sorted_indices = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )
                for i, idx in enumerate(sorted_indices[:20]):
                    token = tokens[idx] if idx < len(tokens) else f"Token_{idx}"
                    score = scores[idx]
                    st.write(f"{i + 1}. '{token}': {score:.4f}")

            with col_bottom:
                st.write("**Top 20 EVALUATION tokens:**")
                sorted_indices_lowest = sorted(
                    range(len(scores)), key=lambda i: scores[i]
                )
                for i, idx in enumerate(sorted_indices_lowest[:20]):
                    token = tokens[idx] if idx < len(tokens) else f"Token_{idx}"
                    score = scores[idx]
                    st.write(f"{i + 1}. '{token}': {score:.4f}")

    with col2:
        # Create colorbar
        if score_type == "probability_class_0":
            min_val, max_val = 0.0, 1.0
        else:
            min_val, max_val = min(scores), max(scores)

        img_colorbar = create_colorbar(min_val, max_val, color_scheme, title)
        st.image(img_colorbar, use_container_width=True)


if __name__ == "__main__":
    main()
