"""Usa Numba cuando está instalado y conserva compatibilidad sin él."""

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(function):
            return function

        return decorate