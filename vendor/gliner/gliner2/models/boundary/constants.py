"""Numeric constants shared by the boundary head."""

# A finite masking sentinel representable by fp16, bf16, and fp32. Several
# proposal terms may be added before masking; using dtype minima can overflow.
MASK_LOGIT = -1.0e4
