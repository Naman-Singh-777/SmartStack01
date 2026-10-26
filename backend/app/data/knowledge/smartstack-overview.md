# What SmartStack does

SmartStack is a small system that lowers the cost of answering questions with language models. It does this by matching the effort to the question instead of using the most expensive model every time.

When a question arrives, SmartStack first runs a trained classifier that labels it simple, moderate or complex and maps that label to a small, medium or large model. If the classifier is unsure, it moves up one tier.

Next it checks a semantic cache. If a very similar question was answered recently, that answer is returned for free.

Otherwise it searches the knowledge base in the backend data folder using hybrid word and fuzzy matching, and sends only the passages that match well. Fewer input tokens means a lower bill.

The chosen model writes the answer. With an Anthropic API key the app calls the real model. Without a key it runs in offline mode and answers by quoting the best matching sentences from the documents.

Finally a compute planner suggests where to run the work: spot machines in the cheapest suitable region for batch work, and on demand machines close to users for interactive work.

Every response reports the estimated cost next to the cost of sending the same request to the large model.
