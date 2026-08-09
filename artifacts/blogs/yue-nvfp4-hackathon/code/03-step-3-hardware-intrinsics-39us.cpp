// Extracted from sources/blogs/yue-nvfp4-hackathon.md by scripts/extract_blog_code.py
// Heading: # Blackwell NVFP4 Kernel Hackathon Journey (Yue Zhang) > ## Key Optimization Steps > ### Step 3: Hardware Intrinsics (~39us)
// Original fence language: cpp
// See artifacts/blogs/yue-nvfp4-hackathon/code/PROVENANCE.yaml for origin + license metadata.

// Generic: manual bit manipulation for FP4 -> FP16 conversion
// float val = decode_fp4_manual(packed_byte >> 4);  // slow

// Hardware intrinsic: single instruction for FP4 -> FP16x2
// __half2_raw result = __nv_cvt_fp4x2_to_halfraw2(packed_fp4, __NV_E2M1);  // fast
