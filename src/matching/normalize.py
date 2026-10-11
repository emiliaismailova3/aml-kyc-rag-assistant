"""Turn a messy company name into a comparable form.

"Xəzər Logistika" MMC  ->  "xezer logistika"
XEZER LOGISTIKA MMC    ->  "xezer logistika"
"""

from __future__ import annotations

import re
import unicodedata

# Legal-form words carry no identity: "X MMC" and "X LLC" are the same business name.
LEGAL_FORMS = {"mmc", "qsc", "asc", "llc", "ltd", "jsc", "oao", "ooo"}

# Azerbaijani letters -> plain Latin. Done by hand because Unicode decomposition
# does not cover them (the schwa "ə" and dotless "ı" have no base letter).
_AZ_TO_LATIN = str.maketrans({"ə": "e", "ğ": "g", "ı": "i", "ö": "o", "ü": "u", "ş": "s", "ç": "c"})


def normalize_name(name: str, strip_legal_form: bool = True) -> str:
    # Handle the Turkic dotted/dotless I *before* lower-casing: Python turns
    # "İ" into "i" + a combining dot, which would otherwise leak into the text.
    name = name.replace("İ", "i").replace("I", "i")
    name = unicodedata.normalize("NFKC", name).lower().translate(_AZ_TO_LATIN)
    # Drop any remaining accents (e.g. a leftover combining dot, or "é").
    name = "".join(c for c in unicodedata.normalize("NFD", name) if not unicodedata.combining(c))
    name = re.sub(r"[^\w\s]", " ", name)  # quotes, dots, dashes, etc.
    words = name.split()
    if strip_legal_form:
        words = [w for w in words if w not in LEGAL_FORMS]
    return " ".join(words)
