def densidad(r, nx, rtope):
    """Perfil radial parabólico de densidad usado para escalar la respuesta."""
    return [2.3e3 * (1 - r[i] / rtope) ** 2 for i in range(nx)]