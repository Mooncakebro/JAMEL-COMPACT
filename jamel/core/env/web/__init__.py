"""
Web 自动化模块
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Optional
import gymnasium as gym
import browsergym.core  # register the openended task as a gym environment
import browsergym.core.env as browsergym_core_env
import browsergym
from browsergym.core.action.highlevel import HighLevelActionSet
import playwright.sync_api
from playwright.sync_api import BrowserType
# import browsergym.webarena  # register webarena tasks as gym environments
# import browsergym.miniwob  # register miniwob tasks as gym environments
from .observer import Observer
from .action_space import get_action_space
from .coverage import start_coverage, save_coverage

from jamel.log import log_utils
logger = log_utils.get_logger(__name__)

browser_timeout = 5000
browser_asset_settle_ms = 1000
browser_asset_timeout_cap_ms = 3000


def _wrap_observation_extractors_for_logging() -> None:
    """Log slow BrowserGym observation phases for diagnosing app-specific hangs."""
    import functools
    import time

    for name in (
        "_pre_extract",
        "extract_dom_snapshot",
        "extract_merged_axtree",
        "extract_focused_element_bid",
        "extract_dom_extra_properties",
        "extract_screenshot",
    ):
        original = getattr(browsergym_core_env, name, None)
        if original is None or getattr(original, "_jamel_logged", False):
            continue

        @functools.wraps(original)
        def wrapped(*args, __name=name, __original=original, **kwargs):
            started = time.monotonic()
            logger.info("BrowserGym observation phase started", phase=__name)
            try:
                return __original(*args, **kwargs)
            finally:
                elapsed = time.monotonic() - started
                logger.info("BrowserGym observation phase finished", phase=__name, elapsed_seconds=round(elapsed, 3))

        wrapped._jamel_logged = True
        setattr(browsergym_core_env, name, wrapped)


_wrap_observation_extractors_for_logging()


original_launch = BrowserType.launch
def patched_launch(self, *args, **kwargs):
    # 这些是解决 Linux/Docker 环境下 Target crashed 的救命参数
    extra_flags = [
        "--no-sandbox", 
        "--disable-setuid-sandbox", 
        "--disable-dev-shm-usage", 
        "--disable-gpu"
    ]
    
    # 拦截 BrowserGym 传过来的参数，偷偷把我们的参数追加进去
    if 'args' in kwargs and kwargs['args'] is not None:
        kwargs['args'] = kwargs['args'] + extra_flags
    else:
        kwargs['args'] = extra_flags
        
    # 放行，调用原本的底层 launch 方法
    return original_launch(self, *args, **kwargs)
# 替换 Playwright 默认的 launch 方法
BrowserType.launch = patched_launch

def wait_for_page_assets(
    page: playwright.sync_api.Page,
    timeout: Optional[int] = None,
    settle_ms: Optional[int] = None,
) -> None:
    timeout = timeout or browser_timeout
    settle_ms = browser_asset_settle_ms if settle_ms is None else settle_ms

    try:
        page.wait_for_load_state("load", timeout=timeout)
    except playwright.sync_api.Error as exc:
        logger.warning("等待页面 load 状态超时或失败", error=str(exc), timeout=timeout)

    try:
        page.wait_for_load_state("networkidle", timeout=timeout)
    except playwright.sync_api.Error as exc:
        logger.warning("等待页面 networkidle 状态超时或失败", error=str(exc), timeout=timeout)

    try:
        page.wait_for_function(
            """
            () => Array.from(document.images || []).every((img) => img.complete)
            """,
            timeout=timeout,
        )
    except playwright.sync_api.Error as exc:
        logger.warning("等待图片 complete 状态超时或失败", error=str(exc), timeout=timeout)

    try:
        page.evaluate(
            """
            async ({ settleMs }) => {
              const withTimeout = (p, ms) => Promise.race([p, new Promise(r => setTimeout(r, ms))]);
              const images = Array.from(document.images || []);
              await Promise.all(images.map((img) => {
                if (typeof img.decode === "function") {
                  return withTimeout(img.decode().catch(() => null), 1000);
                }
                return Promise.resolve();
              }));
              if (document.fonts && document.fonts.ready) {
                await withTimeout(document.fonts.ready.catch(() => null), 2000);
              }
              await new Promise((resolve) => setTimeout(resolve, settleMs));
              await withTimeout(new Promise((resolve) => requestAnimationFrame(() => resolve())), 2000);
            }
            """,
            {"settleMs": settle_ms},
        )
    except playwright.sync_api.Error as exc:
        logger.warning("等待图片 decode/fonts/settle 失败", error=str(exc), timeout=timeout)

def patch_setup_for_open_ended_task(self: browsergym.core.OpenEndedTask, page: playwright.sync_api.Page) -> tuple[str, dict]:
    logger.info(f"setup timeout: {browser_timeout}")
    page.goto(self.start_url, timeout=browser_timeout, wait_until="domcontentloaded")
    wait_for_page_assets(page, timeout=min(browser_timeout, browser_asset_timeout_cap_ms))
    return self.goal, {}

browsergym.core.OpenEndedTask.setup = patch_setup_for_open_ended_task

original_wait_dom_loaded = browsergym_core_env.BrowserEnv._wait_dom_loaded
def patched_wait_dom_loaded(self):
    original_wait_dom_loaded(self)
    for page in list(self.context.pages):
        wait_for_page_assets(page, timeout=min(browser_timeout, browser_asset_timeout_cap_ms))

browsergym_core_env.BrowserEnv._wait_dom_loaded = patched_wait_dom_loaded

@dataclass
class EnvContext:
    env: gym.Env
    obs: Any
    info: Any
    save_step_coverage: Optional[Callable[[os.PathLike, int], None]] = None
    save_final_coverage: Optional[Callable[[os.PathLike], None]] = None

def get_environment(start_url: str, headless: bool, env_id: str="browsergym/openended", record_coverage: bool=False, timeout=600000):
    global browser_timeout; browser_timeout = timeout
    action_mapping = HighLevelActionSet(
        subsets=["chat", "infeas", "bid", "coord", "nav", "tab"],
    ).to_python_code
    env = gym.make(
        env_id,
        task_kwargs={"start_url": start_url},
        wait_for_user_message=False,
        headless=headless,
        timeout=timeout,
        action_mapping=action_mapping,
    )

    obs, info = env.reset()

    cdp_session = None
    if record_coverage:
        page = env.unwrapped.page
        cdp_session = start_coverage(page)

    save_step_coverage = None
    save_final_coverage = None
    if cdp_session:
        def _save_step_coverage(path: os.PathLike, step_index: int):
            logger.info("Web 环境保存步骤 coverage", step_index=step_index, path=path)
            save_coverage(cdp_session, path)

        def _save_final_coverage(path: os.PathLike):
            logger.info("Web 环境保存最终 coverage", path=path)
            save_coverage(cdp_session, path)

        save_step_coverage = _save_step_coverage
        save_final_coverage = _save_final_coverage
    elif record_coverage:
        logger.warning("Web 环境 coverage 已开启但初始化失败，后续不会保存 coverage")
    
    return EnvContext(
        env=env, 
        obs=obs, 
        info=info, 
        save_step_coverage=save_step_coverage,
        save_final_coverage=save_final_coverage,
    )

def stop_envrionment(env_context: EnvContext, record_coverage: bool=False, record_coverage_path: os.PathLike=None):
    if env_context.env:
        if record_coverage and record_coverage_path:
            record_coverage_path = Path(record_coverage_path)
            record_coverage_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Save Coverage info...", record_coverage_path=record_coverage_path)
            if env_context.save_final_coverage:
                env_context.save_final_coverage(record_coverage_path)
            else:
                logger.warning("跳过最终 coverage 保存：环境未提供保存实现", record_coverage_path=record_coverage_path)
        
        env_context.env.close()
        logger.info("Gym 环境已关闭")
        
    else:
        logger.warning("No env found, skip close.")
