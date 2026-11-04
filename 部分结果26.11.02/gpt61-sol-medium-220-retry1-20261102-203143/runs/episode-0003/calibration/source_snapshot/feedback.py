"""Robot-only feedback and explicit receding-prefix execution; no grasp detector."""


def separation(state):
    left, right = state["gripper_joint_position_m"]
    return abs(left - right)


def decision_feedback(obs, step, previous, control_freq):
    result = {"joint_separation_m": separation(obs.state),
              "note": "Joint separation is not contact clearance or proof of grasp. Compare both images before and after a small lift."}
    if previous is None:
        return result
    old_step, old_obs, proposed = previous
    count = step - old_step
    if not 0 < count <= len(proposed):
        raise ValueError("Feedback requires a nonempty executed prefix of the previous decision")
    executed = proposed[:count]
    result["previous_decision"] = {
        "start_step": old_step, "executed_steps": count, "discarded_actions": len(proposed) - count,
        "simulated_seconds": count / control_freq, "executed_actions": executed,
        "eef_delta_world_m": [b - a for a, b in zip(old_obs.state["eef_position_world_m"], obs.state["eef_position_world_m"])],
        "joint_separation_before_m": separation(old_obs.state),
        "joint_separation_change_m": separation(obs.state) - separation(old_obs.state)}
    return result


def feedback_prefix(actions):
    """Stop BEFORE a gripper reversal or switching between arm motion and gripper-only commands.

    At least one validated model action executes. No values are changed or added.
    The discarded suffix is never replayed without a new model decision.
    """
    def gripper_only(a):
        return not any(a[:6]) and a[6] != 0
    direction = actions[0][6]
    previous = actions[0]
    for index, action in enumerate(actions[1:], 1):
        if action[6] and direction and action[6] != direction:
            return index, "gripper_direction_change"
        if ((gripper_only(previous) and any(action[:6])) or
                (any(previous[:6]) and gripper_only(action))):
            return index, "arm_gripper_phase_change"
        # Zero commands must not hide a prior gripper-only -> motion transition.
        if any(action):
            previous = action
        if action[6]:
            direction = action[6]
    return len(actions), None


def gripper_motion_diagnostics(events):
    """Timing evidence only; no assertion about object identity or grasp success."""
    bouts = []
    skipped = 0
    current = None
    direction = -1  # Reset/settling opens the gripper.
    for event in events:
        if event.get("event") != "step":
            continue
        if not all(key in event for key in ("action", "before", "after", "step")):
            skipped += 1
            continue
        action = event["action"]
        if action[6] == 1 and direction != 1:
            current = {"start_step": event["step"], "initial_joint_separation_m": separation(event["before"]),
                       "stationary_close_steps_before_first_arm_motion": 0,
                       "first_arm_motion_step": None, "joint_separation_before_first_arm_motion_m": None}
            bouts.append(current)
        if action[6] == -1:
            current = None
        if action[6]:
            direction = action[6]
        if current is not None:
            current["last_step"] = event["step"]
            current["final_joint_separation_m"] = separation(event["after"])
            if current["first_arm_motion_step"] is None:
                if any(action[:6]):
                    current["first_arm_motion_step"] = event["step"]
                    current["joint_separation_before_first_arm_motion_m"] = separation(event["before"])
                elif action[6] == 1:
                    current["stationary_close_steps_before_first_arm_motion"] += 1
    return {"closing_bouts": bouts, "skipped_incomplete_steps": skipped,
            "note": "Command timing and robot joint separation only; not a grasp success detector."}
