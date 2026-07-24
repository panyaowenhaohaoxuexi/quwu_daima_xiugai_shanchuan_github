import torch

from Eval_EMA import load_model
from model import FogRoutedRGBTIRDehazer


def test_eval_ema_explicitly_selects_requested_checkpoint_state(tmp_path):
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    teacher_state = {key: value.clone() for key, value in model.state_dict().items()}
    student_state = {key: value.clone() for key, value in model.state_dict().items()}
    first_key = next(iter(student_state))
    student_state[first_key] = student_state[first_key] + 1
    path = tmp_path / "ema.pt"
    torch.save({
        "format_version": 1, "training_stage": "ema", "model_class": "FogRoutedRGBTIRDehazer",
        "density_gt_semantics": "transmission",
            "config": {
                "base_channels": 8, "router_hidden_channels": 8,
                "deform_num_samples": 4, "deform_max_offset": 2.0,
                    "num_structure_renderers": 2, "memory_max_tokens": 16,
                    "memory_topk": 2, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
                "memory_reliability_epsilon": 1e-6,
                "memory_reliable_ratio_threshold": 0.01,
                "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
                "boundary_width": 1,
            }, "teacher": teacher_state, "student": student_state,
    }, path)

    teacher, _ = load_model(path, "teacher")
    student, _ = load_model(path, "student")

    assert torch.equal(teacher.state_dict()[first_key], teacher_state[first_key])
    assert torch.equal(student.state_dict()[first_key], student_state[first_key])
