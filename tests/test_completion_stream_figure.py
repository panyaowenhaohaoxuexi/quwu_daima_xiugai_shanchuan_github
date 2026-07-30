from PIL import Image


def test_structured_completion_query_prefers_high_route_with_ir_structure():
    from tools.completion_stream_figure import select_structured_completion_query

    route = __import__("torch").tensor([[[[1.0, 0.95], [0.90, 0.10]]]])
    structure_energy = __import__("torch").tensor([[[[0.01, 0.80], [0.30, 1.00]]]])

    assert select_structured_completion_query(route, structure_energy) == (0, 1)


def test_structured_completion_query_excludes_padding_border():
    from tools.completion_stream_figure import select_structured_completion_query

    torch = __import__("torch")
    route = torch.ones(1, 1, 9, 9)
    structure_energy = torch.zeros(1, 1, 9, 9)
    structure_energy[0, 0, 0, 0] = 10.0
    structure_energy[0, 0, 4, 4] = 1.0

    assert select_structured_completion_query(route, structure_energy, border_fraction=0.2) == (4, 4)


def test_completion_panel_highlights_only_the_top_appearance_source():
    from tools.completion_stream_figure import compose_completion_panel

    full = Image.new("RGB", (640, 512), "#8090a0")
    ir = Image.new("L", (640, 512), 128)
    attention = Image.new("RGB", (640, 512), "#91b8d7")
    output = Image.new("RGB", (640, 512), "#99b787")
    patches = [
        Image.new("RGB", (100, 80), "#c7e3f4"),
        Image.new("RGB", (100, 80), "#e9be74"),
        Image.new("RGB", (100, 80), "#c58b98"),
    ]

    panel = compose_completion_panel(
        hazy=full,
        infrared=ir,
        attention=attention,
        output=output,
        patches=patches,
        weights=[0.50, 0.30, 0.20],
        query_xy=(160, 140),
    )

    assert panel.mode == "RGB"
    assert panel.width >= 1400
    assert panel.height < 600
    assert panel.getpixel((1020, 340))[:2] == (199, 227)
