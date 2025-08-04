# Predicting Post-Earnings Announcement Drift Using NLP

This repo contains the code, data, and modeling notebooks for a project investigating whether earnings call transcripts can be used to predict short-term post-announcement stock price direction (1–5 trading days). The work explores encoder-based (FinBERT), decoder-based (LLaMA-3.1), and ensemble approaches, with a focus on long-only strategies and directional accuracy.

## Repository Structure

### Data and Preprocessing

- **00_Data/**  
  Contains raw and intermediate datasets including transcript metadata, company-ticker mappings, and price data used throughout the project.

- **01_Transcript_Data_Collection_and_Preprocessing.ipynb**  
  Gathers and cleans earnings call transcript data, maps companies to tickers, and deduplicates transcript components by `keydevid`.

- **02_Stock_Price_Collection_and_Preprocessing.ipynb**  
  Collects stock price data using `yfinance`, merges with transcript metadata, and constructs labels for classification.

### Baseline and Analysis

- **03_EDA_and_Baseline_Model_Testing.ipynb**  
  Performs exploratory data analysis (EDA) and evaluates a baseline majority-class classifier for stock drift prediction.

### Experiments and Modeling

- **04_Experiments_Over_Baseline_1.ipynb**  
  Implements FinBERT-based modeling using only the first 512 tokens of each transcript (truncated version). Tests multiple layer-freezing strategies.

- **05_Experiments_Over_Baseline_2.ipynb**  
  Implements a chunked FinBERT + BiLSTM architecture to handle longer transcripts. Also conducts a sentiment audit using FinBERT’s native 3-class classifier to validate optimism bias hypotheses.

- **06_Experiments_Over_Baseline_3.ipynb**  
  Uses Meta’s LLaMA-3.1 decoder-only LLM to generate instruction-based confidence labels (“confident up”, etc.). Constructs a feedforward classifier and tests an ensemble strategy combining LLaMA and FinBERT predictions.

### Model Outputs

- **finbert_senti_results.csv**  
  Contains FinBERT sentiment classifications and prediction results used in evaluation and audit.

- **llama_direction_results.csv**  
  Contains LLaMA-3.1-based confidence predictions and associated results used for ensemble analysis.

### Notes

- Notebooks may not render properly in the GitHub preview pane. **I recommend downloading and opening them locally** (e.g., in Jupyter Notebook or VS Code) for full visibility.

---

For details on methodology, findings, and discussion, please refer to the accompanying report.
