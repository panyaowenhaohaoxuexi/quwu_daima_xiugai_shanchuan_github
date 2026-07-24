"""State transitions for route-supervision availability safeguards."""


def update_empty_omega_streak(streak, *, effective_lambda_route, valid_omega_count, step_succeeded):
    """Advance only for a committed step with active but unavailable route supervision."""
    if not step_succeeded:
        return int(streak)
    if effective_lambda_route <= 0 or valid_omega_count > 0:
        return 0
    return int(streak) + 1
