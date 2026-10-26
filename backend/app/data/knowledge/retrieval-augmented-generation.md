# Retrieval augmented generation

Retrieval augmented generation, usually shortened to RAG, gives a model the right documents at the moment it answers. Instead of hoping the model memorised a fact, you search your own documents, pick the best passages and put them in the prompt.

Documents are first split into chunks, often a paragraph or two each. At question time every chunk gets a score for how well it matches the question, and the top few are kept.

Hybrid search combines two kinds of scoring. Word matching rewards chunks that contain the same words as the question. Fuzzy matching, for example on groups of characters, forgives spelling mistakes and word forms. Adding the two scores together finds relevant chunks that either method alone would miss.

Set a minimum score. If nothing matches well, send no context at all. A weak passage costs tokens and can mislead the model.

Keep chunks short and the number of chunks small. Fewer, better passages mean fewer input tokens, a lower bill and often a more accurate answer.

Embedding models are another way to score meaning, using vectors learned by a neural network. They find matches with no shared words but need more computing power than classic word statistics.
