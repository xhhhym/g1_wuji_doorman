"""Order-independent task terminations; advance() is idempotent per high-level step."""


def _task(env):
    task = env.command_manager.get_term("door_task")
    task.advance()
    return task


def task_invalid_state(env):
    return _task(env).invalid


def task_fall(env):
    return _task(env).fallen


def task_success(env):
    return _task(env).success


def task_stage_timeout(env):
    return _task(env).stage_timeout
