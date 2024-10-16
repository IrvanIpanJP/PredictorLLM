# Memory ID descriptions for each layer
short_memory_id_desc = "The ID of the short-term memory information."
mid_memory_id_desc = "The ID of the mid-term memory information."
long_memory_id_desc = "The ID of the long-term memory information."
reflection_memory_id_desc = "The ID of the reflection-term memory information."

# ID extraction prompts (train / test)
train_memory_id_extract_prompt = (
    "Provide the piece of {memory_layer} memory most relevant to the investment decision "
    "from major investment sources (e.g., ARK, Two Sigma, Bridgewater)."
)

test_memory_id_extract_prompt = (
    "Select the piece of {memory_layer} memory most relevant to today's investment decision."
)

# Summaries
train_trade_reason_summary = (
    "Explain the rationale behind the investment decision based on the provided information."
)
test_trade_reason_summary = (
    "Explain the reason for your investment decision, given the data and recent price movement."
)

test_invest_action_choice = (
    "Based on the information, choose an action: buy, sell, or hold the asset."
)

# Info prefixes
train_investment_info_prefix = (
    "Today's date is {cur_date}. For {symbol}, the next day's price difference is {future_record}.\n\n"
)
test_investment_info_prefix = (
    "The symbol to analyze is {symbol}, and today's date is {cur_date}.\n"
)

# Sentiment and momentum explanations
test_sentiment_explanation = """For example, positive headlines can lift sentiment,
leading to buying and higher prices; negative news often causes selling and lower prices.
Sentiment metrics (positive, neutral, negative) are expressed as proportions summing to 1.
"""

test_momentum_explanation = """Below is a summary of recent price fluctuations (momentum).
Momentum suggests that trends can persist: rising assets may continue rising, and vice versa.
"""

# Guardrails prompts
train_prompt = """You are analyzing historical data. Using the following information, explain why the market behaves as observed from one day to the next, then reference the relevant memory ID(s).

${investment_info}

${gr.complete_json_suffix_v2}
"""

test_prompt = """You are analyzing current market data. Based on the memory information (short/mid/long/reflection) and recent momentum, make a single investment decision (buy, sell, or hold) and provide a brief reason. Include the relevant memory ID(s).

${investment_info}

${gr.complete_json_suffix_v2}

"""
