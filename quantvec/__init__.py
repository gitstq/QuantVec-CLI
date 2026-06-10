"""
QuantVec-CLI: A CLI tool for vector quantization, compression analysis,
and RAG framework adapter generation.

Author: QuantVec Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "QuantVec Team"
__license__ = "MIT"

from quantvec.quantizers.base import BaseQuantizer
from quantvec.quantizers.turboquant import TurboQuantQuantizer
from quantvec.quantizers.scalar import ScalarQuantizer
from quantvec.quantizers.product import ProductQuantizer

__all__ = [
    "BaseQuantizer",
    "TurboQuantQuantizer",
    "ScalarQuantizer",
    "ProductQuantizer",
]
