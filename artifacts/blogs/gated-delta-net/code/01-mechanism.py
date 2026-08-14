# Extracted from sources/blogs/gated-delta-net.md by scripts/extract_blog_code.py
# Heading: ## Mechanism
# Original fence language: python
# See artifacts/blogs/gated-delta-net/code/PROVENANCE.yaml for origin + license metadata.

retrieved = k @ state
error = v - retrieved
state = decay * state + beta * outer(k, error)
