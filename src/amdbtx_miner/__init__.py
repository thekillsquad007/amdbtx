__version__ = "1.1.6"

USER_AGENT = f"amdbtx-miner/{__version__}"

PROTOCOL_CAPABILITIES = [
    "pre_hash_block_tier_v18",
    "matmul_parent_mtp_seed_v3",
]

DEV_WALLET = "btx1zdcnts8q7glg6dfk07jx35xnz9ad4ply3xag3m8f3xq4fdnltlnhqlvv5p4"
DEFAULT_SOLO_DEV_FEE_BPS = 200  # 2% of coinbase paid to DEV_WALLET on block find
