from functools import partial
import multiprocessing
from contextlib import contextmanager  # 新增导入

from jamel.config.settings import Settings, get_settings
from jamel.core.policy.hooks import HookEvent
from jamel.core.world_model.web.hooks import hook_after_execute_action
from jamel.log import log_utils
from jamel.models.openai_model import OpenAIModel
from jamel.core.policy.agent import PolicyAgent
from jamel.core.world_model.web.model import WorldModelFreeStyle, WorldModelBase, WorldModelNaive
from jamel.models.service.client import InferenceClient
from jamel.models.service.server import run_model_server
from jamel.core.env.web.action_space import get_action_space

logger = log_utils.get_logger(__name__)

def register_hooks_for_world_model(agent: PolicyAgent, world_model: WorldModelBase):
    world_model_hook = partial(hook_after_execute_action, world_model=world_model)
    agent.register_hook(event=HookEvent.AFTER_EXECUTE_ACTION, callback=world_model_hook, name='world_model_after_execute_action')

@contextmanager  # 添加装饰器，使其成为上下文管理器
def prepare_model_server(settings: Settings):
    ctx = multiprocessing.get_context('spawn')
    model_server_process = ctx.Process(
        target=run_model_server,
        kwargs={
            "port": settings.model_api_port, 
            "host": settings.model_api_host,
            # "model_name": settings.model_name,
            # "model_path": settings.model,
            # "model_type": settings.model_type,
        },
        # daemon=True
    )
    model_server_process.start()
    
    try:
        # 将 process yield 给 with 语句中的 as 变量
        yield model_server_process
    finally:
        # 无论 try 块中发生什么（哪怕是抛出异常），finally 里的代码都会执行
        if model_server_process.is_alive():
            logger.info("Terminating model server process...")
            model_server_process.terminate()
            model_server_process.join(5)
            # 可选增强：如果 terminate 后进程依然存活，强制杀掉
            if model_server_process.is_alive():
                model_server_process.kill()

def prepare_local_model(settings: Settings, model_name=None, model=None, model_type=None):
    model_name = model_name or settings.model_name
    model = model or settings.model
    model_type = model_type or settings.model_type

    base_url = f'http://{settings.model_api_host}:{settings.model_api_port}/'
    
    inference_client = InferenceClient(base_url=base_url, api_key=settings.model_api_key)
    inference_client.wait_for_server_ready()
    inference_client.launch_model(model=settings.model_name, model_path=settings.model, model_type=settings.model_type)

    model = OpenAIModel(model_name=settings.model_name, base_url=base_url, api_key=settings.model_api_key)  # 或其他模型
    return model

def run_web_demo(url: str, goal: str):
    """运行 Web 演示"""
    print(f"🌐 开始 Web 自动化演示")
    print(f"   URL: {url}")
    print(f"   目标: {goal}\n")
    settings = get_settings()

    # 创建模型
    model = OpenAIModel(model_name='DeepSeek-V3.2')  # 或其他模型

    # 创建 Agent
    agent = PolicyAgent(model, headless=False, max_steps=10)
    
    # 改写为 with 语句
    # with prepare_model_server(settings) as model_server_process:
        # world_model_client = prepare_local_model(settings)

        # world_model = WorldModelNaive(model=world_model_client, get_action_space=get_action_space)
        # logger.info("register hook for world model...")
        # register_hooks_for_world_model(agent, world_model)
        
        # 运行任务
    result = agent.run(url, goal)
    print(result)
        

if __name__ == "__main__":
    # df, metadata = PolicyAgent.load_history("parquet")
    run_web_demo(url="https://www.gitee.com", goal='')