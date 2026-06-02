import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import io
import multiprocessing
import threading
import traceback
from typing import Dict
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn
from argparse import ArgumentParser, Namespace

from jamel.utils import communication_utils
from jamel.utils.general_utils import summarize_str
from jamel.utils.communication_utils import get_channel_multiprocessing, get_channel_queue

from .base import ModelInterface
from jamel.log import log_utils

logger = log_utils.get_logger(__name__)
# ----------------------------------------------------------------------
# Parse arguments
parser = ArgumentParser(allow_abbrev=False)
parser.add_argument('--host', type=str, required=False, default="0.0.0.0", help='Service Host')
parser.add_argument('--port', type=int, required=False, default=8888, help='Service Port')
parser.add_argument('--model', type=str, required=False, default=None, help='Served model name')
parser.add_argument('--model_path', type=str, required=False, default=None, help='Path to the model directory')
parser.add_argument('--model_type', type=str, required=False, default=None, help='Model Type')
# ----------------------------------------------------------------------
# Load Model Logic using dictionary mapping
# ----------------------------------------------------------------------
MODEL_TYPE_MAPPING = {
    # Base models
    'qwen2_base': 'Qwen2BaseCompletionModel',
    'qwen2_5_base': 'Qwen2BaseCompletionModel',
    'qwen3_base': 'Qwen2BaseCompletionModel',
    'qwen2_vl_base': 'Qwen2VLBaseCompletionModel',
    'qwen2_5_vl_base': 'Qwen2VLBaseCompletionModel',
    'qwen3_vl_base': 'Qwen2VLBaseCompletionModel',

    # VL models
    'qwen2_vl': 'Qwen2VLModel',
    'qwen2_5_vl': 'Qwen2VLModel',
    
    # text models
    'qwen2': 'Qwen2Model',
    'qwen2_5': 'Qwen2Model',
    'qwen3': 'Qwen3Model',

    # Default
    'default': 'Qwen2Model'
}

def _get_model_class(model_path: str, model_type: str) -> ModelInterface:
    """根据模型路径和类型返回模型类名"""
    def _get_model_class_name():
        # 优先使用 model_type
        if model_type and model_type in MODEL_TYPE_MAPPING:
            return MODEL_TYPE_MAPPING[model_type]

        # 从路径推断
        model_path_lower = model_path.lower()

        if 'base' in model_path_lower:
            if 'vl' in model_path_lower:
                return 'Qwen2VLBaseCompletionModel'
            else:
                return 'Qwen2BaseCompletionModel'
        elif 'vl' in model_path_lower:
            return 'Qwen2VLModel'

        # 默认模型
        return MODEL_TYPE_MAPPING['default']
    model_class_name = _get_model_class_name()
    # 动态导入对应的模型类
    if model_class_name == 'Qwen2VLBaseCompletionModel':
        from .utils import Qwen2VLBaseCompletionModel
        PlanModel = Qwen2VLBaseCompletionModel
    elif model_class_name == 'Qwen2BaseCompletionModel':
        from .utils import Qwen2BaseCompletionModel
        PlanModel = Qwen2BaseCompletionModel
    elif model_class_name == 'Qwen2VLModel':
        from .utils import Qwen2VLModel
        PlanModel = Qwen2VLModel
    elif model_class_name == 'Qwen2Model':
        from .utils import Qwen2Model
        PlanModel = Qwen2Model
    elif model_class_name == 'Qwen3Model':
        from .utils import Qwen3Model
        PlanModel = Qwen3Model
    else:
        from .utils import Qwen2Model
        PlanModel = Qwen2Model
    return PlanModel


# ==========================================
# FastAPI Setup
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动时的逻辑 ---
    # 如果需要在启动 API 前就加载模型，可以在这里调用
    if args.model_path is not None:
        logger.info(f"Auto-launching model from {args.model_path}...")
        try:
            launch_model_core(
                model=args.model or args.model_path,
                model_path=args.model_path,
                model_type=args.model_type,
            )
        except Exception as e:
            logger.error(f"Failed to launch initial model: {e}")
            # 这里可以选择是否直接 sys.exit(1)
    
    yield  # 应用运行中...
    # --- 关闭时的逻辑 (Ctrl+C 后会执行这里) ---
    logger.info("Shutting down model workers...")
    for model_id, online_process in online_process_dict.items():
        if online_process.process.is_alive():
            logger.info(f"Terminating {model_id}...")
            online_process.writer.put(None)
            # proc_obj.process.terminate()
            # proc_obj.process.join(timeout=2) # 等待退出
            if online_process.process.is_alive():
                online_process.process.kill() # 强制杀死
    online_process_dict.clear()
    logger.info("All models terminated.")

app = FastAPI(lifespan=lifespan)
router = APIRouter() # 创建一个 Router

# Helper functions
def base64_to_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    return image

# Data Models
class MultiModalPromptRequest(BaseModel):
    prompt: str
    base64_images: list[str]

class PromptRequest(BaseModel):
    prompt: str

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    base64_images: list[str] = []
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class LaunchModelRequest(BaseModel):
    model_path: str
    model_type: str = None
    model: str = None
    kwargs: dict = field(default_factory=dict)

class TerminateModelRequest(BaseModel):
    model: str = None
    model_path: str = None

def model_worker(model, model_path, model_type, channel, **kwargs):
    """
    这个函数会在一个完全独立的进程中运行。
    """
    try:
        reader, writer = channel
        model_cls = _get_model_class(model_path, model_type)
        logger.info(f"model class: {model_cls.__name__}", model_path=model_path, model_type=model_type)
        model_instance = model_cls(model_path, **kwargs)
        logger.info("【子进程】模型加载完成，等待任务...")
        writer.put({"status": "success", "return_value": model})
    except Exception as e:
        traceback.print_exc()
        writer.put({
            "status": "error", "return_value": {
                "error": str(e), 
                "traceback": traceback.format_exc() # 【慎用】完整的堆栈信息
            }
        })
        return
    while True:
        try:
            # 1. 从主进程接收数据
            payload = reader.get() 
            if payload is None:
                break
            method_name, method_kwargs = payload
            
            logger.info(f"【子进程】正在调用: {method_name}, 参数：{method_kwargs}")
            # 2. 执行推理 (模拟)
            # result = model.generate(prompt)
            response = getattr(model_instance, method_name)(**method_kwargs)

            # 3. 把结果发回主进程
            writer.put({"status": "success", "return_value": response})
            
        except Exception as e:
            traceback.print_exc()
            writer.put({
                "status": "error", "return_value": {
                    "error": str(e), 
                    "traceback": traceback.format_exc() # 【慎用】完整的堆栈信息
                }
            })

@dataclass
class OnlineProcess:
    process: multiprocessing.Process
    reader: multiprocessing.Queue
    writer: multiprocessing.Queue
    channel_name: str
    model: str
    model_path: str
    model_type: str
    request_lock: threading.Lock = field(default_factory=threading.Lock)

online_process_dict: Dict[str, OnlineProcess] = {}
# ------------------------------------------
# Define Routes on the Router
# ------------------------------------------

# Model management API endpoints:
# - launch_model: POST /v1/models - Launch a new model
# - [Unimplemented] get_launch_model_progress: GET /v1/models/{model_id}/progress - Get loading progress
# - [Unimplemented] cancel_launch_model: POST /v1/models/{model_id}/cancel - Cancel model loading
# - terminate_model: DELETE /v1/models or DELETE /v1/models/{model_id} - Terminate a model
# - list_models: GET /v1/models - List all loaded models

@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    # 打印堆栈信息到后台日志（方便后端看）
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            # "message": str(exc),  # 错误的具体简述 (例如: "division by zero")
            "traceback": traceback.format_exc() # 【慎用】完整的堆栈信息
        },
    )

def launch_model_core(model: str, model_path: str, model_type: str, **kwargs):
    channel_name = f"inference_{model}"
    reader, writer = get_channel_multiprocessing(channel_name, channel_id=0)
    # 启动子进程
    model_process = multiprocessing.Process(
        target=model_worker, 
        args=(model, model_path, model_type, get_channel_multiprocessing(channel_name=channel_name, channel_id=1)),
        kwargs=kwargs,
        daemon=True
    )
    model_process.start()
    result = reader.get()
    if result['status'] == 'error':
        raise ChildProcessError(result['return_value'])
    online_process_dict[model] = OnlineProcess(
        process=model_process,
        channel_name=channel_name,
        model=model,
        model_path=model_path,
        model_type=model_type,
        reader=reader,
        writer=writer,
    )

@router.post("/models")
def launch_model(request: LaunchModelRequest):
    """Launch a new model"""
    model = request.model or request.model_path
    
    logger.info(f"收到加载模型请求 - Model: {model}, Path: {request.model_path}, Type: {request.model_type}")
    # 【修复】这里之前是 online_model_dict (它是空的)，改为检查 online_process_dict
    if model in online_process_dict:
        logger.info(f"模型 {model} 已经在运行中，跳过加载。")
        return {
            "model_id": model,
            "model_path": request.model_path,
            "model_type": request.model_type,
            "status": "launched (already running)"
        }
    
    try:
        launch_model_core(model, request.model_path, request.model_type, **request.kwargs)
        logger.info(f"模型 {model} 加载并启动成功。")
    except Exception as e:
        logger.error(f"模型 {model} 启动失败: {e}")
        raise e
    return {
        "model_id": model,
        "model_path": request.model_path,
        "model_type": request.model_type,
        "status": "launched"
    }

@router.delete("/models/{model_id:path}")
def terminate_model(model_id: str):
    """Terminate a model by ID"""
    logger.info(f"收到卸载模型请求 - Model ID: {model_id}")
    
    if model_id not in online_process_dict:
        logger.warning(f"卸载失败：模型 {model_id} 未找到或未运行。")
        return JSONResponse(status_code=404, content={"error": f"Model {model_id} not found"})
    try:
        online_process = online_process_dict[model_id]
        if online_process.process.is_alive():
            online_process.writer.put(None)
        if online_process.process.is_alive():
            online_process.process.kill()
        # online_process.process.join(timeout=60)
        
        # 清理资源
        if online_process.channel_name in communication_utils.channels:
            del communication_utils.channels[online_process.channel_name]
        del online_process_dict[model_id]
        
        logger.info(f"模型 {model_id} 已成功卸载。")
    except Exception as e:
        logger.error(f"卸载模型 {model_id} 时发生错误: {e}")
        raise e
    return {
        "model_id": model_id,
        "status": "terminated"
    }

@router.post("/chat/completions")
async def get_chat_response(request: dict):
    """
    Standard OpenAI chat completion endpoint.
    Handles requests to /chat/completions AND /v1/chat/completions
    """
    model = request.get('model')
    messages = request.get('messages', [])
    
    # 简单的日志记录，避免打印过长的 content
    msg_count = len(messages)
    logger.info(f"收到 Chat 请求 - Model: {model}, Messages: {msg_count} 条, Request: {request}")
    if model not in online_process_dict:
        logger.error(f"请求失败：模型 {model} 未加载。当前可用模型: {list(online_process_dict.keys())}")
        return JSONResponse(status_code=404, content={
            "error": "Model not found", 
            "message": f"The model '{model}' does not exist."
        })
    online_process = online_process_dict.get(model)
    
    # 当前实现每个在线模型只有一个 worker 进程和一对共享队列。
    # 这里显式串行化请求，避免并发请求时读到彼此的响应。
    with online_process.request_lock:
        online_process.writer.put((ModelInterface.get_chat_response.__name__, request))
        result = online_process.reader.get()
    
    if result['status'] == 'error':
        logger.error(f"模型 {model} 推理过程中发生错误: {result['return_value']}")
        raise ChildProcessError(result['return_value'])
    
    response = result['return_value']
    
    # 计算 token 使用量用于日志 (如果返回结构里有 usage)
    usage_info = response.get("usage", {})
    logger.info(f"Chat 请求完成 - Model: {model}, Usage: {usage_info}")
    return response

@router.get("/models") 
def list_models():
    logger.info("收到列出所有模型请求 (List Models)")
    models_list = [
        {
            "id": online_process.model,
            "object": "model",
            "created": 1677610602,
            "owned_by": "organization-owner",
            "model_path": online_process.model_path,
            "model_type": online_process.model_type
        } for online_process in online_process_dict.values()
    ]
    logger.info(f"当前运行模型数量: {len(models_list)}")
    return {
        "object": "list",
        "data": models_list,
    }

@router.get("/models") # 通常 OpenAI 客户端会请求这个端点
def list_models():
    # OpenAI 格式通常返回一个列表，这里做简单的兼容
    return {
        "object": "list",
        "data": [
            {
                "id": online_process.model,
                "object": "model",
                "created": 1677610602,
                "owned_by": "organization-owner",
                "model_path": online_process.model_path,
                "model_type": online_process.model_type
            } for online_process in online_process_dict.values()
        ],
    }

@router.get("/health")
def health_check():
    return {"status": "healthy"}

# ------------------------------------------
# Mount Router
# ------------------------------------------

# 1. 挂载到 /v1 前缀 (例如: /v1/chat/completions)
app.include_router(router, prefix="/v1")

# 2. 同时也挂载到根目录 (例如: /chat/completions)
# 这样旧的客户端代码不需要修改也能工作
app.include_router(router)

def run_model_server(**kwargs):
    global args
    args, _ = parser.parse_known_args(namespace=Namespace(**kwargs))
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    run_model_server()
