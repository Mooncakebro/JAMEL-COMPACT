"""
Hook System for PolicyAgent - 灵活的事件订阅机制
"""
from __future__ import annotations
from enum import Enum
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from jamel.core.policy.brains.utils import Decision
from jamel.log import log_utils

logger = log_utils.get_logger(__name__)


class HookEvent(Enum):
    """Hook 事件类型"""
    # 生命周期事件
    BEFORE_RUN = "before_run"  # 运行开始前
    AFTER_RUN = "after_run"    # 运行结束后

    # 步骤事件
    BEFORE_STEP = "before_step"        # 每步执行前
    AFTER_STEP = "after_step"          # 每步执行后

    # 决策事件
    BEFORE_DECIDE_ACTION = "before_decide_action"  # 决策前 (world model 触发点)
    AFTER_DECIDE_ACTION = "after_decide_action"    # 决策后

    # 执行事件
    BEFORE_EXECUTE_ACTION = "before_execute_action"  # 执行动作前
    AFTER_EXECUTE_ACTION = "after_execute_action"    # 执行动作后

    # 奖励和记忆事件
    BEFORE_REWARD = "before_reward"          # 计算奖励前
    AFTER_REWARD = "after_reward"            # 计算奖励后
    BEFORE_MEMORY_UPDATE = "before_memory_update"  # 内存更新前
    AFTER_MEMORY_UPDATE = "after_memory_update"    # 内存更新后

    # 错误处理
    ON_ERROR = "on_error"  # 发生错误时

@dataclass
class HookData:
    before_obs: Any = None
    after_obs: Any = None
    before_observation: str = None
    after_observation: str = None
    before_info: Any = None
    after_info: Any = None
    decision: Decision = None
    raw_reward: float = None # reward from gym
    reward: float = None # reward calculated by us
    terminated: bool = None
    truncated: bool = None
    cummulated_reward: float = None


@dataclass
class HookContext:
    """
    Hook 上下文，传递给 hook 回调函数
    目前的设计缺陷明显，每次的 context 都是重新构造，我们应该让 data 字段固定下来，在所有任务之间保持同一个引用才对。
    """
    event: HookEvent
    agent: Any  # PolicyAgent 实例
    data: HookData = None
    result: Optional[Any] = None
    error: Optional[Exception] = None

    def update(self, **kwargs):
        """更新上下文数据"""
        self.data.update(kwargs)
        return self


@dataclass
class Hook:
    """Hook 定义"""
    event: HookEvent
    callback: Callable[[HookContext], Optional[HookContext]]
    priority: int = 0  # 优先级，数字越小越先执行
    name: Optional[str] = None  # Hook 名称，用于调试和移除

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.event.value}_{self.callback.__name__}"


class HookManager:
    """
    Hook 管理器
    管理所有注册的 hooks 并触发相应事件
    """

    def __init__(self):
        self._hooks: Dict[HookEvent, List[Hook]] = {
            event: [] for event in HookEvent
        }

    def register(
        self,
        event: HookEvent,
        callback: Callable[[HookContext], Optional[HookContext]],
        priority: int = 0,
        name: Optional[str] = None
    ) -> Hook:
        """
        注册 hook

        Args:
            event: 事件类型
            callback: 回调函数，接收 HookContext，返回修改后的 HookContext 或 None
            priority: 优先级，数字越小越先执行
            name: Hook 名称

        Returns:
            Hook 实例
        """
        hook = Hook(event=event, callback=callback, priority=priority, name=name)
        self._hooks[event].append(hook)
        # 按优先级排序
        self._hooks[event].sort(key=lambda h: h.priority)

        logger.debug(
            "Hook registered",
            event_value=event.value,
            hook_name=hook.name,
            priority=priority
        )
        return hook

    def unregister(self, hook: Hook) -> bool:
        """
        注销 hook

        Args:
            hook: 要移除的 Hook 实例

        Returns:
            是否成功移除
        """
        if hook.event in self._hooks and hook in self._hooks[hook.event]:
            self._hooks[hook.event].remove(hook)
            logger.debug("Hook unregistered", hook_name=hook.name)
            return True
        return False

    def unregister_by_name(self, name: str, event: Optional[HookEvent] = None) -> int:
        """
        通过名称注销 hooks

        Args:
            name: Hook 名称
            event: 可选的事件类型，如果为 None 则在所有事件中查找

        Returns:
            移除的 hook 数量
        """
        count = 0
        events = [event] if event else list(HookEvent)

        for evt in events:
            hooks_to_remove = [h for h in self._hooks[evt] if h.name == name]
            for hook in hooks_to_remove:
                self._hooks[evt].remove(hook)
                count += 1

        if count > 0:
            logger.debug("Hooks unregistered by name", name=name, count=count)
        return count

    def trigger(self, context: HookContext) -> HookContext:
        """
        触发事件

        Args:
            context: Hook 上下文

        Returns:
            修改后的上下文
        """
        hooks = self._hooks.get(context.event, [])

        if not hooks:
            return context

        logger.debug(
            "Triggering hooks",
            event_value=context.event.value,
            hook_count=len(hooks)
        )

        for hook in hooks:
            try:
                result = hook.callback(context)
                if result is not None:
                    context = result
            except Exception as e:
                logger.error(
                    "Hook execution failed",
                    hook_name=hook.name,
                    event_value=context.event.value,
                    error=str(e),
                    exc_info=True
                )
                # 可以选择继续执行其他 hooks 或抛出异常
                # 这里选择记录错误但继续执行

        return context

    def clear(self, event: Optional[HookEvent] = None):
        """
        清除 hooks

        Args:
            event: 可选的事件类型，如果为 None 则清除所有
        """
        if event:
            self._hooks[event] = []
        else:
            for evt in HookEvent:
                self._hooks[evt] = []
        logger.debug("Hooks cleared", event_value=event.value if event else "all")

    def get_hooks(self, event: HookEvent) -> List[Hook]:
        """获取某个事件的所有 hooks"""
        return self._hooks[event].copy()


def hook_decorator(event: HookEvent, priority: int = 0, name: Optional[str] = None):
    """
    装饰器方式注册 hook

    Usage:
        @hook_decorator(HookEvent.BEFORE_DECIDE_ACTION, priority=10)
        def my_hook(context: HookContext):
            # Do something
            return context
    """
    def decorator(func: Callable[[HookContext], Optional[HookContext]]):
        func._hook_event = event
        func._hook_priority = priority
        func._hook_name = name or f"{event.value}_{func.__name__}"
        return func
    return decorator
