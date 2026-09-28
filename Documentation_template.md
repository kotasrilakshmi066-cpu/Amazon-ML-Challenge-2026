# Business Entity Resolution Challenge - Methodology

## 1. Methodology Used
Our approach leverages a highly scalable, rule-based entity resolution pipeline designed specifically to maximize Precision for the F_0.5 metric. Rather than relying on a machine learning classifier that might overfit to class imbalances in heuristically blocked training data, we utilized a strict deterministic blocking strategy followed by hard token-set overlap thresholds. 

## 2. Candidate Generation / Blocking Strategy
To meet the requirement of scaling across billions of records while strictly controlling memory limits, we used a highly precise deterministic block key:
- **Block Key Extraction**: We isolated the first full alphanumeric word of every `business_name`.
- **Noise Reduction**: Generic words (e.g., "National", "United") that appeared more than 50 times in the reference source were dropped to prevent combinatorial (Cartesian) explosion.
- **Candidate Join**: We performed an exact-match database join between Source 1 and Sources 2/3 on `[country, block_key]`. 

## 3. Model Architecture and Feature Engineering
- **Feature Engineering**: For the generated candidate pairs, we utilized the `rapidfuzz` library to compute `fuzz.token_set_ratio` for both names and addresses. This metric is permutation-invariant and ignores extra generic words, natively handling abbreviations and DBA/trade name variations (e.g., "Apple" vs "Apple Computer Inc").
- **Classification Rules**: Instead of a probabilistic model, we applied strict hard thresholds: `name_token_set_ratio > 85` AND `addr_token_set_ratio > 80`. This heavily restricts false positives, directly optimizing the precision-biased F_0.5 score.

## 4. Other Relevant Information
- **Performance Optimization**: Python's native `pandas.apply()` was bypassed during feature engineering. We extracted columns to native lists and utilized list comprehensions `[fuzz.token_set_ratio(n1, n2) for n1, n2 in zip(...)]`, vastly improving throughput.
- **Dependencies**: The pipeline relies entirely on `pandas`, `numpy`, and `rapidfuzz` (no scikit-learn or XGBoost required).
