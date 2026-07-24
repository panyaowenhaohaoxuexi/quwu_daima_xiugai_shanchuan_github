from training.omega_state import update_empty_omega_streak


def test_empty_omega_streak_only_advances_after_successful_enabled_route_step():
    assert update_empty_omega_streak(5, effective_lambda_route=0.0, valid_omega_count=0, step_succeeded=True) == 0
    assert update_empty_omega_streak(5, effective_lambda_route=1.0, valid_omega_count=1, step_succeeded=True) == 0
    assert update_empty_omega_streak(5, effective_lambda_route=1.0, valid_omega_count=0, step_succeeded=False) == 5
    assert update_empty_omega_streak(5, effective_lambda_route=1.0, valid_omega_count=0, step_succeeded=True) == 6
