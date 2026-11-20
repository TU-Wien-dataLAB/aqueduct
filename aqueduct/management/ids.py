MAX_ID = 2**63 - 2
_MAX_DIGITS = len(str(MAX_ID))


def parse_id(raw: str | None) -> int | None:
    if not (raw and len(raw) <= _MAX_DIGITS and raw.isascii() and raw.isdigit()):
        return None
    if int(raw) > MAX_ID:
        return None
    return int(raw)
