"""
Compatibility shims for third-party library issues.

Import this module before importing vLLM or transformers to apply patches.
"""

# transformers 5.x removed `all_special_tokens_extended` from the tokenizers
# backend, but vLLM still references it. This adds a no-op property so vLLM
# can load tokenizers without crashing.
# Tracking: https://github.com/huggingface/transformers/issues/xxxxx
try:
    import transformers.tokenization_utils_tokenizers as tut
    if not hasattr(tut.TokenizersBackend, "all_special_tokens_extended"):
        tut.TokenizersBackend.all_special_tokens_extended = property(lambda self: [])
except (ImportError, AttributeError):
    pass
