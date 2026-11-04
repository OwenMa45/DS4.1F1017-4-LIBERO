"""Fixed diagnostic selection from LIBERO_COMMIT, task_order_index=0."""

from .config import Config
from .errors import ExperimentError


FIVE_TASKS = (
    ("libero_spatial", 0, "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"),
    ("libero_object", 0, "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
    ("libero_goal", 0, "open_the_middle_drawer_of_the_cabinet"),
    ("libero_goal", 5, "push_the_plate_to_the_front_of_the_stove"),
    ("libero_10", 5, "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"),
)


def five_configs(base):
    if base.protocol_version != "ds-libero-direct-v2":
        raise ExperimentError("Five-task diagnostics require a v2 profile")
    return [Config.from_dict({**base.to_dict(), "evaluation_scope": "diagnostic",
        "suite": suite, "task_id": task, "expected_task_name": name,
        "task_order_index": 0, "init_state_id": 0, "seed": 0})
        for suite, task, name in FIVE_TASKS]
