# Business Entity Resolution Pipeline

## Instructions to Reproduce End-to-End

1. **Environment Setup**:
   Ensure you have Python 3.9+ installed. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Placement**:
   Ensure the `dataset/` folder is placed in the root directory alongside this code, containing the `train/` and `test/` subdirectories.

3. **Execution**:
   Run the baseline pipeline from the root directory:
   ```bash
   python code/business_entity_resolution/src/baseline.py
   ```

4. **Output**:
   The script will automatically generate the `output/` directory containing:
   - `candidate_pairs.tsv`
   - `matching_results.tsv`
