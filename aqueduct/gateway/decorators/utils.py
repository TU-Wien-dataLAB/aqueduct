def in_wildcard(value: str | None, allowed_values: list[str]) -> bool:
    """Check if a value is in a list of allowed values or matches a wildcard pattern."""
    if value is None:
        return False

    valid = value in allowed_values
    if not valid:
        # Check wildcard port patterns (e.g., "http://localhost:*")
        for allowed in allowed_values:
            if allowed.endswith(":*"):
                base_origin = allowed[:-2]
                if value.startswith(base_origin + ":"):
                    return True
    return valid
