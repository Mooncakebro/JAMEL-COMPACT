
def split_fields(raw_string: str, field_list: list[str]):
    split_by_text = lambda s, t: (lambda i : (s[:i], s[i + len(t):]))(s.index(t))
    remaining = raw_string
    result_dict = {}
    for field_idx, (field_name, field_prefix) in list(enumerate(field_list))[::-1]:
        if field_prefix in remaining:
            remaining, field_str = split_by_text(remaining, "" if field_idx == 0 else '\n' + field_prefix.strip())
            result_dict[field_name] = field_str.strip().removeprefix(field_prefix) if field_idx == 0 else field_str.strip()
        else:
            result_dict[field_name] = ''
    return result_dict

if __name__ == "__main__":
    raw_content = '''
--- Thought ---
Hello here is thought.
--- Action ---
Hi there is action.
'''
    field_list = [("thought", "--- Thought ---"), ("action", "--- Action ---"), ("test", "--- test ---")]
    parsed_fields = split_fields(raw_content, field_list)
    print(parsed_fields)