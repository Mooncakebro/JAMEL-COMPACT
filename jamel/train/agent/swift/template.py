import copy
from swift.template import register_template, TEMPLATE_MAPPING

base_type = 'qwen3' 

if base_type in TEMPLATE_MAPPING:
    memory_template_meta = copy.deepcopy(TEMPLATE_MAPPING[base_type])
    memory_template_meta.template_type = f'{base_type}_memory' # 预测的最后一个 token 总是 --- Thought ---!
    
    memory_template_meta.suffix = ['--- Thought ---']
    
    register_template(memory_template_meta)

    policy_template_meta = copy.deepcopy(TEMPLATE_MAPPING[base_type])
    policy_template_meta.template_type = f'{base_type}_policy'
    
    register_template(policy_template_meta)