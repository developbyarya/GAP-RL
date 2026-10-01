import re

def strip_custom_sac():
    with open('gap_rl/algorithms/scripts/custom_sac.py', 'r') as f:
        content = f.read()

    # 1. Remove TokenSelfAttention class
    match = re.search(r'class TokenSelfAttention\(nn\.Module\):.*?(?=\n# ============================================================================\n# 2\. Actor)', content, re.DOTALL)
    if match:
        content = content.replace(match.group(0), "")

    # 2. Rename use_attn_lstm to use_lstm and remove d_attn
    content = content.replace("use_attn_lstm", "use_lstm")
    content = re.sub(r',\s*d_attn:\s*int\s*=\s*384', '', content)
    content = re.sub(r'self\.d_attn\s*=\s*d_attn\n', '', content)
    content = re.sub(r'd_attn=self\.d_attn,?\n?', '', content)
    content = re.sub(r'd_attn=args\.d_attn,?\n?', '', content)
    content = re.sub(r'd_attn=d_attn,?\n?', '', content)

    # 3. Remove token_attn initialization
    content = re.sub(r'\s*self\.token_attn = TokenSelfAttention\(.*?use_lstm\)', '', content, flags=re.DOTALL)
    
    # 4. Fix lstm_input_dim
    content = re.sub(r'lstm_input_dim = 5 \* .*? else features_dim', 'lstm_input_dim = features_dim', content)

    # 5. Remove token_attn call from _extract_windowed_features
    content = re.sub(r'\s*flat_feat = self\.token_attn\(flat_obs_pre, flat_feat\)\n', '\n', content)

    with open('gap_rl/algorithms/scripts/custom_sac.py', 'w') as f:
        f.write(content)

def strip_sac_train():
    with open('gap_rl/algorithms/scripts/sac_train.py', 'r') as f:
        content = f.read()
    
    content = content.replace("use_attn_lstm", "use_lstm")
    content = content.replace("use-attn-lstm", "use-lstm")
    content = content.replace("attn", "lstm")
    
    # Remove d_attn argparse
    match = re.search(r'\s*parser\.add_argument\(\s*"--d-attn".*?\)', content, re.DOTALL)
    if match:
        content = content.replace(match.group(0), "")
        
    content = re.sub(r'\s*d_attn=args\.d_attn,', '', content)

    with open('gap_rl/algorithms/scripts/sac_train.py', 'w') as f:
        f.write(content)

strip_custom_sac()
strip_sac_train()
print("Stripped successfully")
