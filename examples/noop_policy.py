"""Executable adapter example. Holds the arm and keeps the gripper open.

This is an interface smoke policy, NOT a task-solving baseline. No API calls.
Run it through the real pinned LIBERO launcher as documented in docs/microbench.md.
"""

from ds_libero.microbench import EpisodeContext, PolicyInput


class NoopPolicy:
    def act(self, request: PolicyInput) -> list[list[float]]:
        # A real adapter decodes request.observation.images, combines the task
        # instruction/state/history, and converts its model output to this format.
        return [[0., 0., 0., 0., 0., 0., -1.] for _ in range(request.max_actions)]


def create_policy(context: EpisodeContext) -> NoopPolicy:
    # Called once per episode. Load/reset model session state here; non-secret
    # checkpoint/prompt/settings can be passed through context.policy_config.
    return NoopPolicy()
