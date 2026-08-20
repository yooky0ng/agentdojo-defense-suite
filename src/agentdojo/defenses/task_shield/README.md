# Task Shield

This directory contains a minimal independent implementation of *The Task Shield:
Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents*
(Jia et al., ACL 2025). No official implementation was publicly available.

The implementation keeps only the paper's core path:

1. extract and store user task instructions;
2. check assistant natural-language instructions;
3. block misaligned outgoing tool calls;
4. inspect incoming tool outputs and attach the paper's feedback;
5. treat an aggregate `ContributeTo` score of zero as misaligned.

The extraction, content-checking, tool-call-checking, and feedback prompts are transcribed
from Appendix E. The paper does not specify a feedback retry limit or JSON parse failure
policy. This implementation uses two feedback rounds and allows an item when the defender
returns no parseable score. Conversation history is serialized as text and supplied to the
checker because the paper requires contextual scoring but does not publish its exact
serialization format.
