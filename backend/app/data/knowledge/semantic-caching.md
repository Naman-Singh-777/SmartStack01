# Semantic caching

A normal cache only helps when the exact same text arrives again. A semantic cache also recognises questions that mean the same thing, such as "how do I reset my password" and "password reset steps". When it finds a match it returns the stored answer, which costs nothing and takes a few milliseconds.

The danger is a false hit. Two questions can look almost identical and still need different answers, like the refund policy for the US and the refund policy for the EU. A careful semantic cache therefore checks more than text similarity. For example, it can also require that the important words match.

Set an expiry time on cached answers. Facts change, and an answer that was right last week may be wrong today. A one hour lifetime is a reasonable starting point for general questions, and a shorter one suits news or prices.

Track the hit rate. If only a few percent of requests hit the cache, the extra complexity may not be worth it for your traffic. If repeated questions are common, such as a support bot, the savings can be large.

Never share cached answers between users when the answer depends on private data.
