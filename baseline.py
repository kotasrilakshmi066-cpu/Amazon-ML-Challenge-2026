import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from xgboost import XGBClassifier
import os
import re

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading data...")
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t").fillna("")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t").fillna("")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t").fillna("")
ground_truth = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t").fillna("")

s23 = pd.concat([s2, s3], ignore_index=True)

# ==========================================
# 2. THE 0.336 BLOCKING STRATEGY (Exact Revert)
# ==========================================
def get_block_key(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]', '', text) 
    # EXACTLY the 6-character rule that got your best score
    return text[:6] 

print("Creating block keys...")
s1['block_key'] = s1['business_name'].apply(get_block_key)
s23['block_key'] = s23['business_name'].apply(get_block_key)

s1 = s1[s1['block_key'].str.len() >= 4] 
s23 = s23[s23['block_key'].str.len() >= 4]

# EXACTLY the safe memory limit that got your best score
s1_counts = s1['block_key'].value_counts()
valid_keys = s1_counts[s1_counts < 50].index 
s1 = s1[s1['block_key'].isin(valid_keys)]
s23 = s23[s23['block_key'].isin(valid_keys)]

print("Generating candidate pairs...")
train_candidates = pd.merge(
    s1[['entity_id', 'country', 'block_key']], 
    s23[['entity_id', 'country', 'block_key']], 
    on=['country', 'block_key'], 
    suffixes=('_s1', '_s23')
)
train_candidates = train_candidates.rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_s23': 'matched_entity_id'})
print(f"Generated {len(train_candidates)} candidate pairs for training.")

# ==========================================
# 3. FEATURE ENGINEERING (Adding Partial Ratio for an extra boost!)
# ==========================================
print("Extracting features...")
def extract_features(pairs_df, s1_df, s23_df):
    if len(pairs_df) == 0: return pd.DataFrame()
        
    df = pairs_df.merge(s1_df[['entity_id', 'business_name', 'business_address']], 
                        left_on='source1_entity_id', right_on='entity_id')
    df = df.merge(s23_df[['entity_id', 'business_name', 'business_address']], 
                  left_on='matched_entity_id', right_on='entity_id', suffixes=('_1', '_2'))
    
    names_1 = df['business_name_1'].astype(str).str.lower().tolist()
    names_2 = df['business_name_2'].astype(str).str.lower().tolist()
    addrs_1 = df['business_address_1'].astype(str).str.lower().tolist()
    addrs_2 = df['business_address_2'].astype(str).str.lower().tolist()
    
    print("  -> Calculating Similarities...")
    df['name_ratio'] = [fuzz.ratio(n1, n2) for n1, n2 in zip(names_1, names_2)]
    df['addr_ratio'] = [fuzz.ratio(a1, a2) for a1, a2 in zip(addrs_1, addrs_2)]
    df['name_set_ratio'] = [fuzz.token_set_ratio(n1, n2) for n1, n2 in zip(names_1, names_2)]
    df['addr_set_ratio'] = [fuzz.token_set_ratio(a1, a2) for a1, a2 in zip(addrs_1, addrs_2)]
    
    # NEW: We keep this magic feature from the last run to help the AI be slightly smarter than the 0.336 run
    df['name_partial'] = [fuzz.partial_ratio(n1, n2) for n1, n2 in zip(names_1, names_2)]
    
    return df[['source1_entity_id', 'matched_entity_id', 'name_ratio', 'addr_ratio', 'name_set_ratio', 'addr_set_ratio', 'name_partial']]

original_s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t").fillna("")
X_train_pairs = extract_features(train_candidates, original_s1, s23)

print("Mapping ground truth...")
gt_pairs = set()
for _, row in ground_truth.iterrows():
    s1_id = row['source1_entity_id']
    matches = str(row['matched_entity_ids']).split(',')
    for m in matches:
        if m.strip(): gt_pairs.add((s1_id, m.strip()))

X_train_pairs['is_match'] = [1 if (s1, m) in gt_pairs else 0 for s1, m in zip(X_train_pairs['source1_entity_id'], X_train_pairs['matched_entity_id'])]

# ==========================================
# 4. TRAIN MODEL
# ==========================================
print("Training XGBoost classifier...")
features = ['name_ratio', 'addr_ratio', 'name_set_ratio', 'addr_set_ratio', 'name_partial']

model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
if len(X_train_pairs) > 0:
    model.fit(X_train_pairs[features], X_train_pairs['is_match'])

# ==========================================
# 5. PREDICT ON TEST SET
# ==========================================
print("Processing Test Set...")
t1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t").fillna("")
t2 = pd.read_csv("dataset/test/test_source2.tsv", sep="\t").fillna("")
t3 = pd.read_csv("dataset/test/test_source3.tsv", sep="\t").fillna("")
t23 = pd.concat([t2, t3], ignore_index=True)

t1['block_key'] = t1['business_name'].apply(get_block_key)
t23['block_key'] = t23['business_name'].apply(get_block_key)

t1 = t1[t1['block_key'].str.len() >= 4]
t23 = t23[t23['block_key'].str.len() >= 4]

t1_counts = t1['block_key'].value_counts()
valid_t1_keys = t1_counts[t1_counts < 50].index
t1 = t1[t1['block_key'].isin(valid_t1_keys)]
t23 = t23[t23['block_key'].isin(valid_t1_keys)]

test_candidates = pd.merge(
    t1[['entity_id', 'country', 'block_key']], 
    t23[['entity_id', 'country', 'block_key']], 
    on=['country', 'block_key'], 
    suffixes=('_s1', '_s23')
).rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_s23': 'matched_entity_id'})

original_t1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t").fillna("")
X_test_pairs = extract_features(test_candidates, original_t1, t23)

if len(X_test_pairs) > 0:
    X_test_pairs['match_prob'] = model.predict_proba(X_test_pairs[features])[:, 1]
    # EXACT same threshold that got you 0.336
    THRESHOLD = 0.65 
    final_predictions = X_test_pairs[X_test_pairs['match_prob'] > THRESHOLD]
else:
    final_predictions = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_id'])

# ==========================================
# 6. FORMAT SUBMISSION
# ==========================================
print("Formatting submission files...")
os.makedirs("output", exist_ok=True)

if len(test_candidates) > 0:
    candidate_grouped = test_candidates.groupby('source1_entity_id')['matched_entity_id'].apply(list).reset_index()
    candidate_grouped['candidate_entity_ids'] = candidate_grouped['matched_entity_id'].apply(lambda x: ",".join(x))
else:
    candidate_grouped = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_ids'])

submission_candidates = pd.DataFrame({'source1_entity_id': original_t1['entity_id']})
submission_candidates = submission_candidates.merge(candidate_grouped[['source1_entity_id', 'candidate_entity_ids']], on='source1_entity_id', how='left').fillna("")
submission_candidates.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)

if len(final_predictions) > 0:
    match_grouped = final_predictions.groupby('source1_entity_id')['matched_entity_id'].apply(list).reset_index()
    match_grouped['matched_entity_ids'] = match_grouped['matched_entity_id'].apply(lambda x: ",".join(x))
else:
    match_grouped = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])

submission_matches = pd.DataFrame({'source1_entity_id': original_t1['entity_id']})
submission_matches = submission_matches.merge(match_grouped[['source1_entity_id', 'matched_entity_ids']], on='source1_entity_id', how='left').fillna("")
submission_matches.to_csv("output/matching_results.tsv", sep="\t", index=False)

print("Done! Files saved to output/ folder.")
