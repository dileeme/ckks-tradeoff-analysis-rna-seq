import tenseal as ts

def create_context(poly_mod_degree):
    """
    CKKS context for tradeoff analysis across three modulus levels.
    Scale is fixed at 2**30 across all levels for comparable results.

    Depth budget per level:
    - 4096 : depth 1  (add, scalar mul only — no ciphertext-ciphertext mul)
    - 8192 : depth 3  (dot product + shallow ops)
    - 16384: depth 5  (dot product + z-score normalization)
    """

    if poly_mod_degree == 4096:
        # Total coeff bits must stay under ~109 bits for 4096
        # [first, middle, last] — middle primes = scale bits = 30
        coeffs = [35, 30, 35]
        scale  = 2**30

    elif poly_mod_degree == 8192:
        # Total coeff bits must stay under ~218 bits for 8192
        coeffs = [35, 30, 30, 30, 35]
        scale  = 2**30

    elif poly_mod_degree == 16384:
        # Total coeff bits must stay under ~438 bits for 16384
        coeffs = [40, 30, 30, 30, 30, 30, 40]
        scale  = 2**30

    else:
        raise ValueError(f"Unsupported poly_mod_degree: {poly_mod_degree}. Use 4096, 8192, or 16384.")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=coeffs
    )
    context.global_scale = scale
    context.generate_galois_keys()
    context.generate_relin_keys()

    return context