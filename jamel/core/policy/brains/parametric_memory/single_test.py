input_prompt = '''
You are an intelligent autonomous explorer. 
Your goal is to conduct a systematic and engaging exploration of the environment, uncovering hidden details and interesting patterns. Try your best to discover new things.

--- Valid Actions ---

Note: This action set allows you to interact with your environment. Most of them are python function executing playwright code. The primary way of referring to elements in the page is through bid which are specified in your observations. 20 different types of actions are available.
noop(wait_ms: float = 1000)
send_msg_to_user(text: str)
report_infeasible(reason: str)
scroll(delta_x: float, delta_y: float)
fill(bid: str, value: str)
select_option(bid: str, options: str | list[str])
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
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

For example, a valid action is:
click('12')


--- Current Observation ---

Last Action: click('644')

Last Action Result: Success.

Current open pages URLs:
('http://localhost:8000/agoda/',)

Current open pages titles:
('Agoda Concept',)

Current active page index:
[0]

Current Observation: 
RootWebArea 'Agoda Concept', focused
	[603] button '\uf060'
		StaticText '\uf060'
	[605] heading 'Confirm Booking'
	[608] image ''
	[610] heading 'Shinjuku Granbell Hotel'
	[611] paragraph ''
		StaticText 'Tokyo, Japan'
	StaticText 'Standard Room'
	StaticText 'Nov 15 - 18'
	StaticText '3 Nights'
	StaticText '2 Adults, 0 Children'
	[618] heading 'Contact Details'
	[621] LabelText ''
		StaticText 'Contact Name'
	[622] textbox '' value='Alex Traveler'
		StaticText 'Alex Traveler'
	[625] LabelText ''
		StaticText 'Email'
	[626] textbox '' value='alex@example.com'
		StaticText 'alex@example.com'
	[628] LabelText ''
		StaticText 'Nights'
	[629] button '3 \uf078'
		StaticText '3'
		StaticText '\uf078'
	[634] LabelText ''
		StaticText 'Adults'
	[635] button '2 \uf078'
		StaticText '2'
		StaticText '\uf078'
	[639] LabelText ''
		StaticText 'Children'
	[640] button '0 \uf078'
		StaticText '0'
		StaticText '\uf078'
	[644] button 'Traveler Information \uf078', focused
		StaticText 'Traveler Information'
		StaticText '\uf078'
	[657] button 'Special Requests \uf078'
		StaticText 'Special Requests'
		StaticText '\uf078'
	[671] heading 'Promo Code'
	[672] button '\uf145 Select a Coupon'
		StaticText '\uf145'
	[675] heading 'Payment Method'
	StaticText '\uf1f0'
	StaticText 'Visa â¢â¢â¢â¢ 4242'
	StaticText 'Expires 12/25'
	StaticText '\uf058'
	[683] button 'Change Payment Method'
	[685] heading 'Price Details'
	StaticText '3 Nights x $145'
	StaticText '$435'
	StaticText 'Taxes & Fees'
	StaticText '$43'
	StaticText 'Total'
	StaticText '$478'
	StaticText '\uf101'
	StaticText 'Slide to Pay $478'
	[386] navigation ''
		[388] button '\uf015 Home'
			StaticText '\uf015'
			StaticText 'Home'
		[391] button '\uf0f2 Trips'
			StaticText '\uf0f2'
			StaticText 'Trips'
		[394] button '\uf02b Deals'
			StaticText '\uf02b'
			StaticText 'Deals'
		[397] button '\uf004 Saved'
			StaticText '\uf004'
			StaticText 'Saved'
		[400] button '\uf007 Account'
			StaticText '\uf007'
			StaticText 'Account'


--- Instructions ---

You must respond strictly using the format below. Do not add any text outside these sections.

--- Memory ---
(Recall relevant experiences from memory.)

--- Thought ---
(Think step-by-step. Analyze the current observation, recall your memory, and determine the most strategic next move.)

--- Action ---
(Choose exactly one action from the Valid Actions provided above.)

Now give your response based on all the requirements above.

'''

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = os.getenv("QWEN3_8B_MODEL_PATH", "Qwen/Qwen3-8B")
CHAT_TEMPLATE_PATH = "jamel/models/chat_template/qwen3_continue.jinja"


ACTION_BLOCK_RE = re.compile(r"--- Action ---\s*(.*)", re.S)


def extract_action(response_text: str) -> str:
    match = ACTION_BLOCK_RE.search(response_text or "")
    if not match:
        return ""
    block = match.group(1).strip()
    line = next((ln.strip() for ln in block.splitlines() if ln.strip()), "")
    return line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of samples")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--out", type=str, default="outputs/single_test_results.json")
    return parser.parse_args()


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def apply_chat_template(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def generate_once(model, tokenizer, messages, max_tokens, temperature, top_p, device):
    text = apply_chat_template(tokenizer, messages)
    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
        )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def generate_with_format_control(model, tokenizer, user_prompt, max_tokens, temperature, top_p, device):
    fields = [
        ("memory", "--- Memory ---"),
        ("thought", "--- Thought ---"),
        ("action", "--- Action ---"),
    ]
    full_content = ""

    for _, tag in fields:
        if tag not in full_content:
            current_prefix = (full_content + "\n" + tag).strip()
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": current_prefix},
            ]
            new_part = generate_once(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )
            if new_part.startswith(current_prefix):
                full_content = new_part
            else:
                full_content = current_prefix + new_part
    return full_content


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.chat_template = Path(CHAT_TEMPLATE_PATH).read_text(encoding="utf-8")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        device_map="auto" if args.device is None else None,
    )
    if args.device is not None:
        model = model.to(args.device)
    device = args.device or model.device

    results = []
    for i in range(args.n):
        decoded = generate_with_format_control(
            model=model,
            tokenizer=tokenizer,
            user_prompt=input_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        action = extract_action(decoded)
        action_type = action.split("(")[0] if "(" in action else action
        results.append(
            {
                "index": i + 1,
                "action": action,
                "action_type": action_type,
                "response": decoded,
            }
        )
        print(f"[{i + 1}] {action}")

    action_set = sorted({r["action"] for r in results if r["action"]})
    action_type_set = sorted({r["action_type"] for r in results if r["action_type"]})

    summary = {
        "total_samples": len(results),
        "unique_actions": len(action_set),
        "unique_action_types": len(action_type_set),
        "actions": action_set,
        "action_types": action_type_set,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "chat_template": CHAT_TEMPLATE_PATH,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
