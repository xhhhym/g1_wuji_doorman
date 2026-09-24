"""Termination ordering is explicit: invalid_state advances the task first."""


def task_invalid_state(env):
    task = env.command_manager.get_term("door_task")
    task.advance()
    return task.invalid


def task_fall(env):
    return env.command_manager.get_term("door_task").fallen


def task_success(env):
    return env.command_manager.get_term("door_task").success


def task_stage_timeout(env):
    return env.command_manager.get_term("door_task").stage_timeout
