JAMEL_ACTION_SPACE = """
noop(wait_ms: float = 1000)
send_msg_to_user(text: str)
report_infeasible(reason: str)
scroll(delta_x: float, delta_y: float)
fill(bid: str, value: str)
select_option(bid: str, options: str | list[str])
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
hover(bid: str)
press(bid: str, key_comb: str)
focus(bid: str)
clear(bid: str)
drag_and_drop(from_bid: str, to_bid: str)
upload_file(bid: str, file: str | list[str])
tab_close()
tab_focus(index: int)
new_tab()
go_back()
go_forward()
goto(url: str)
"""

JAMEL_TEMPLATE_NO_HIS = """
You are an autonomous browser exploration agent.
Your exploration goal is: {task_description}
You should interact with the target website to maximize novel JavaScript execution coverage.
Current target URL: {target_url}

The browser action space is:
{action_space}

Your current observation is:
{current_observation}

The current webpage screenshot is:
<image>

Now take exactly one browser action for the current step.
You must reason inside <think> </think> tags first, then output one exact action inside <action> </action> tags.
"""

JAMEL_TEMPLATE = """
You are an autonomous browser exploration agent.
Your exploration goal is: {task_description}
You should interact with the target website to maximize novel JavaScript execution coverage.
Current target URL: {target_url}

You have already taken {step_count} step(s). The most recent {history_length} observations and actions are:
{action_history}

The browser action space is:
{action_space}

You are now at step {current_step}. Your current observation is:
{current_observation}

The current webpage screenshot is:
<image>

Now take exactly one browser action for the current step.
You must reason inside <think> </think> tags first, then output one exact action inside <action> </action> tags.
"""
