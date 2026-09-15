"""Google HTTP errors safe to include in CLI output and durable diagnostics."""


def request(session, method, url, *, service, error, **kwargs):
    try:
        return getattr(session, method)(url, **kwargs)
    except Exception:
        # Transport exceptions can contain request URLs, headers and payloads.
        detail = (
            "A write may have succeeded. Check the destination before retrying."
            if method != "get" else "Check the connection and Google authorization."
        )
        raise error(f"{service} could not be reached. {detail}") from None


def response_json(response, operation, *, service, error):
    try:
        response.raise_for_status()
    except Exception:
        status = getattr(response, "status_code", None)
        detail = f" (HTTP {status})" if type(status) is int else ""
        raise error(
            f"{service} {operation} failed{detail}. "
            "Check resource access and Google authorization."
        ) from None
    try:
        value = response.json()
    except Exception:
        raise error(f"{service} {operation} returned invalid JSON.") from None
    if not isinstance(value, dict):
        raise error(f"{service} {operation} returned an invalid response.")
    return value
