# How language model pricing works

Language model providers charge by the token. A token is a piece of a word, and English text averages roughly four characters per token. Input tokens (the prompt you send) and output tokens (the answer the model writes) are priced separately, and output tokens usually cost several times more than input tokens.

Larger models cost much more per token than smaller ones. A top tier model can cost ten times or more than a small model from the same family, even when both give the same answer to an easy question. This gap is the reason routing exists: most questions people ask do not need the largest model.

Prompt length matters too. Every extra paragraph of context is billed as input on every request. Sending only the few passages that actually relate to the question keeps the bill down, and it often makes the answer better because the model has less noise to read.

To estimate the cost of a request, multiply input tokens by the input price, multiply output tokens by the output price, and add the two. Prices change often, so always read your provider's current pricing page.
