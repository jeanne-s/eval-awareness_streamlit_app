# Insider Trading Token Score Visualization

This Streamlit app visualizes token-level scores from probe projections for insider trading detection. It allows you to explore how different tokens in the insider trading conversations are scored by the probe model.

## Features

- **Interactive Token Visualization**: Color-coded tokens based on their scores
- **Multiple Score Types**: Visualize either projection scores or class 0 probabilities
- **Text Highlighting**: Highlight high-scoring tokens in the original text
- **Statistical Analysis**: View score distributions and statistics
- **Sample Selection**: Browse through different samples in the dataset
- **Original Messages**: View the underlying conversation messages

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_streamlit.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_insider_trading_viz.py
```

2. The app will open in your browser at `http://localhost:8501`

3. Use the sidebar controls to:
   - Select different samples from the dataset
   - Choose between "projections" or "probability_class_0" scores
   - Select different color schemes for visualization
   - Adjust the highlight threshold for high-scoring tokens

## Data Requirements

The app expects the following files to be present:

- `output_data/probe_projections/QwQ-32B/58/insider_trading.jsonl` - The probe projection results
- `data/insider_trading/insider_trading.json` - The original insider trading dataset

## Understanding the Visualizations

### Token Score Visualization
- Each token is displayed as a colored box
- Colors indicate the score value (higher scores = different colors based on the selected color scheme)
- The visualization shows the token-level scores across the entire conversation

### Text with Highlighted Tokens
- Shows the original text with high-scoring tokens highlighted in yellow
- Use the threshold slider to adjust which tokens are considered "high-scoring"

### Score Distribution
- Histogram showing the distribution of scores across all tokens
- Helps identify patterns in the scoring

### Statistics
- Mean, standard deviation, min, max, and median of the scores
- Summary of high-scoring tokens

## File Structure

```
├── streamlit_insider_trading_viz.py    # Main Streamlit app
├── requirements_streamlit.txt          # Python dependencies
├── README_streamlit.md                 # This file
├── output_data/
│   └── probe_projections/
│       └── QwQ-32B/
│           └── 58/
│               └── insider_trading.jsonl
└── data/
    └── insider_trading/
        └── insider_trading.json
```

## Troubleshooting

- **File not found errors**: Make sure the JSONL and dataset files are in the correct locations
- **Tokenization errors**: The app uses the same tokenizer as the original analysis. If you encounter issues, check that the tokenizer model is available
- **Memory issues**: For very large datasets, consider processing smaller batches or using a machine with more RAM

## Customization

You can modify the app by:

1. **Changing the model**: Update the `model_name` parameter in `load_tokenizer()`
2. **Adding new visualizations**: Extend the app with additional Plotly charts
3. **Modifying color schemes**: Add new color schemes to the sidebar options
4. **Adding new analysis**: Implement additional statistical analysis functions 