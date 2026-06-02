import multiprocessing
from jamel.config.settings import Settings, get_settings
from jamel.core.env.web.observer import Observer
from jamel.core.world_model.web.model import WorldModelFreeStyle, WorldModelNaive
from jamel.log import log_utils
from jamel.models.openai_model import OpenAIModel
from jamel.core.policy.agent import PolicyAgent
from jamel.models.service.client import InferenceClient
from jamel.models.service.server import run_model_server
import gymnasium as gym

logger = log_utils.get_logger(__name__)


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
    return model_server_process

def prepare_local_model(settings: Settings, model_name=None, model=None, model_type=None):
    model_name = model_name or settings.model_name
    model = model or settings.model
    model_type = model_type or settings.model_type

    prepare_model_server(settings)
    base_url = f'http://{settings.model_api_host}:{settings.model_api_port}/'
    
    inference_client = InferenceClient(base_url=base_url, api_key=settings.model_api_key)
    inference_client.wait_for_server_ready()
    inference_client.launch_model(model=settings.model_name, model_path=settings.model, model_type=settings.model_type)

    model = OpenAIModel(model_name=settings.model_name, base_url=base_url, api_key=settings.model_api_key)  # 或其他模型
    return model

def run_world_model_demo():
    settings = get_settings()
    
    # 1. 创建 gym 环境
    env = gym.make(
        "browsergym/openended",
        task_kwargs={"start_url": "https://www.gitee.com"},
        wait_for_user_message=False,
        headless=True
    )
    obs, info = env.reset()
    test_obs = Observer.get_observation(obs)
    logger.info("model name: ", settings.model_name)
    base_url = f'http://{settings.model_api_host}:{settings.model_api_port}/'
    model_server_process = prepare_model_server(settings)
    
    inference_client = InferenceClient(base_url=base_url, api_key=settings.model_api_key)
    inference_client.wait_for_server_ready()
    inference_client.launch_model(model=settings.model_name, model_path=settings.model, model_type=settings.model_type)

    model = OpenAIModel(model_name=settings.model_name, base_url=base_url, api_key=settings.model_api_key)  # 或其他模型
    # response = model.get_chat_response([make_user('hello!How are you?').serialize()])
    # print(response)

    logger.info(f"observation: \n{test_obs}")
    from jamel.core.env.web.action_space import get_action_space
    world_model = WorldModelNaive(model=model, get_action_space=get_action_space)
    world_model.predict(observation=test_obs, action='click(\'1\')')

    model_server_process.terminate()
    model_server_process.join(5)

if __name__ == "__main__":
    run_world_model_demo()
