# Model routing

Model routing means choosing which model answers each request instead of sending everything to one model. The router looks at the question, estimates how hard it is, and picks the cheapest model that is likely to handle it well.

A simple router uses tiers. Easy questions such as definitions, short facts and formatting go to a small model. Explanations, short code and summaries go to a medium model. Multi step reasoning, system design and long analysis go to a large model.

A router can be wrong. When the classifier is unsure, a safe rule is to escalate one tier up, because paying a little more is better than returning a weak answer. This is sometimes called a confidence threshold.

Good routing is measured, not assumed. Keep a set of example questions with known good answers, run them through each tier, and check where the smaller models fall short. Retrain or retune the router when you see mistakes.

The router itself must be cheap. A small machine learning model such as TF-IDF with logistic regression answers in about a millisecond and costs nothing per request, unlike asking another large model to judge the question.
