import pytest

from training.source import linear_anneal, source_route_schedule


def test_route_schedule_linearly_anneals_temperature_and_gt_teacher_gate():
    start = source_route_schedule(0, tau_start=1.0, tau_end=0.2,
                                  temperature_anneal_steps=10, teacher_anneal_steps=8)
    middle = source_route_schedule(5, tau_start=1.0, tau_end=0.2,
                                   temperature_anneal_steps=10, teacher_anneal_steps=8)
    end = source_route_schedule(20, tau_start=1.0, tau_end=0.2,
                                temperature_anneal_steps=10, teacher_anneal_steps=8)
    assert start == {"route_temperature": 1.0, "teacher_gate_alpha": 1.0}
    assert middle["route_temperature"] == pytest.approx(0.6)
    assert middle["teacher_gate_alpha"] == pytest.approx(0.375)
    assert end == {"route_temperature": 0.2, "teacher_gate_alpha": 0.0}
    assert linear_anneal(1.0, 0.2, 5, 10) == pytest.approx(0.6)


def test_zero_teacher_anneal_steps_uses_predicted_route_from_the_first_step():
    schedule = source_route_schedule(0, tau_start=1.0, tau_end=0.2,
                                     temperature_anneal_steps=10, teacher_anneal_steps=0)
    assert schedule["teacher_gate_alpha"] == 0.0
