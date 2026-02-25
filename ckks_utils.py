import tenseal as ts

def create_context(poly_mod_degree):

    if poly_mod_degree == 4096:
        coeffs = [40, 30, 30]
        scale = 2**20
    elif poly_mod_degree == 8192:
        coeffs = [60, 40, 40, 60]
        scale = 2**20
    elif poly_mod_degree == 16384:
        coeffs = [60, 40, 40, 40, 60]
        scale = 2**40
    else:
        raise ValueError("Unsupported poly modulus degree")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=coeffs
    )

    context.global_scale = 2**40
    context.generate_galois_keys()

    return context