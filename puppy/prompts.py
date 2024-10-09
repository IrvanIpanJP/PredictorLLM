# memory ids
short_memory_id_desc = "The id of the short-term information."
mid_memory_id_desc = "The id of the mid-term information."
long_memory_id_desc = "The id of the long-term information."
reflection_memory_id_desc = "The id of the reflection-term information."

train_memory_id_extract_prompt = (
    "Provide the piece of information related the most to the investment decisions "
    "from mainstream sources such as the investment suggestions of major fund firms like ARK, "
    "Two Sigma, Bridgewater Associates, etc., in the {memory_layer} memory?"
)

test_memory_id_extract_prompt = (
    "Provide the piece of information most relevant to your investment decisions "
    "in the {memory_layer} memory?"
)

# trade summary
train_trade_reason_summary = (
    "Given a professional trader's suggestion, explain why the trader decided this with the information provided."
)
test_trade_reason_summary = (
    "Given the textual data and the summary of the asset's price movement, please explain the reason for your investment decision."
)

test_invest_action_choice = (
    "Given the information, please make an investment decision: buy the asset, sell it, or hold."
)

# investment info
train_investment_info_prefix = (
    "The current date is {cur_date}. Here are the observed financial market facts: "
    "for {symbol}, the price difference between the next trading day and the current trading day is: {future_record}\n\n"
)

test_investment_info_prefix = (
    "The symbol (stock or crypto) to be analyzed is {symbol}, and the current date is {cur_date}."
)

test_sentiment_explanation = """For example, positive news about a company or crypto project can lift investor sentiment, 
encouraging more buying activity which in turn can push asset prices higher. Conversely, negative news can dampen sentiment, 
leading to selling pressure and a decrease in asset prices. News about competitors or similar projects can also have a ripple effect. 
For instance, if a competitor announces a groundbreaking new product, other assets in that sector might see their prices fall 
as investors anticipate a loss of market share.

The positive, neutral, and negative scores are sentiment ratios for proportions of text that fall into each category 
(all summing to 1). These metrics can be used to analyze how sentiment is conveyed in rhetoric for any given sentence.
"""

test_momentum_explanation = """The information below provides a summary of the asset’s price fluctuations over the previous few days, 
which is the 'Momentum' of the asset. Momentum is based on the idea that assets which have performed well in the past 
will continue to perform well, and conversely, assets that have performed poorly will continue to perform poorly.
"""

# prompts
train_prompt = """Given the following information, can you explain why the financial market fluctuation from the current day to the next day behaves like this?
Please provide a concise summary and the ID of the memory item that supports your summary.

${investment_info}

${gr.complete_json_suffix_v2}
"""

test_prompt = """Given the information, can you make an investment decision?
- Consider the short-term, mid-term, long-term, and reflection-level information.
- Consider the momentum of the historical asset price.
- Note that you can switch between risk-seeking and risk-averse tendencies:
  - If cumulative return is positive, you are risk-seeking.
  - If cumulative return is negative, you are risk-averse.
- Consider how many shares/units of the asset are currently held.
- You should provide exactly one of the following investment decisions: buy, sell, or hold.
- If it's really hard to decide between buy or sell, choose hold.
- You also need to provide the ID of the information that supports your decision.

${investment_info}

${gr.complete_json_suffix_v2}
"""
