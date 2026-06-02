import json
import time
from playwright.sync_api import sync_playwright
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import playwright
    import playwright.sync_api

from jamel.log import log_utils
logger = log_utils.get_logger(__name__)

# ==========================================
# 1. 定义覆盖率辅助函数 (Sync API)
# ==========================================
def start_coverage(page: 'playwright.sync_api.Page'):
    """在该页面上建立 CDP 连接并开启覆盖率"""
    try:
        cdp = page.context.new_cdp_session(page)
        cdp.send("Profiler.enable")
        cdp.send("Debugger.enable")
        cdp.send("Profiler.startPreciseCoverage", {
            "callCount": False, 
            "detailed": True
        })
        logger.info("✅ [Coverage] 覆盖率采集已开启")
        return cdp
    except Exception as e:
        logger.error(f"❌ [Coverage] 开启失败: {e}")
        return None

def save_coverage(cdp: 'playwright.sync_api.CDPSession', filename="coverage_data.json"):
    """提取数据，下载源码并保存 (不中断采集)"""
    if not cdp:
        logger.warning("❌ [Coverage] 保存跳过：CDP session 不存在", filename=str(filename))
        return

    logger.info(f"📦 [Coverage] 正在提取数据并保存至 {filename} ...")

    import signal as _sig

    def _timeout_handler(signum, frame):
        raise TimeoutError("CDP call timed out")

    def _cdp_send_with_timeout(method, params=None, timeout=30):
        """Send CDP command with SIGALRM timeout (main thread only)."""
        old = _sig.signal(_sig.SIGALRM, _timeout_handler)
        _sig.alarm(timeout)
        try:
            return cdp.send(method, params) if params else cdp.send(method)
        finally:
            _sig.alarm(0)
            _sig.signal(_sig.SIGALRM, old)

    try:
        # 1. 获取覆盖率原始数据 (注意：此处不要调用 stopPreciseCoverage)
        data = _cdp_send_with_timeout("Profiler.takePreciseCoverage", timeout=30)
      
        script_coverages = data.get('result', [])
        final_data = []
      
        # 2. 遍历下载源码
        logger.info(f"   - 正在下载 {len(script_coverages)} 个脚本的源码...")
        for script in script_coverages:
            url = script['url']
            if not url or not url.startswith('http'):
                continue
              
            try:
                source_obj = _cdp_send_with_timeout("Debugger.getScriptSource", {"scriptId": script['scriptId']}, timeout=10)
                script_source = source_obj.get('scriptSource', '')
              
                final_data.append({
                    "url": url,
                    "scriptId": script['scriptId'],
                    "source": script_source,
                    "functions": script['functions']
                })
            except Exception:
                # 忽略那些已经销毁无法获取源码的脚本
                pass

        # 3. 保存文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False)
        logger.info(f"✅ [Coverage] 数据快照已保存到: {filename}")
      
    except Exception as e:
        logger.error(f"❌ [Coverage] 保存失败: {e}")

def run_manual_test():
    with sync_playwright() as p:
        # 启动有界面的浏览器
        browser = p.chromium.launch(headless=False) 
        page = browser.new_page()
      
        # 1. 打开你本地 host 的网站
        page.goto("http://localhost:8000/cainiao/")
      
        # 2. 开启覆盖率
        cdp = start_coverage(page)
      
        # 3. 开启可持续记录的交互式命令循环
        print("\n" + "="*50)
        print("【持续记录模式已开启】请在浏览器中进行手动操作。")
        print(" - 直接按【回车键】：保存当前覆盖率快照，并继续记录。")
        print(" - 输入【q】并按回车：保存最后一次快照，并退出程序。")
        print("="*50 + "\n")
      
        save_count = 1
        while True:
            user_input = input(f"\n[{save_count}] 等待操作 (回车=保存快照并继续, q=保存并退出): ")
            
            # 使用时间戳生成不同文件，避免互相覆盖。Monocart 报告生成器能一次性读取全部文件并合并
            current_filename = f"coverage_data_{int(time.time())}.json"
            
            if user_input.strip().lower() == 'q':
                logger.info("👋 准备退出，正在保存最后一份数据...")
                save_coverage(cdp, current_filename)
                
                # 在彻底退出前可以优雅地关闭
                try:
                    cdp.send("Profiler.stopPreciseCoverage")
                except Exception:
                    pass
                break
            else:
                # 没有输入q，仅仅按了回车，只拉取数据不退出
                save_coverage(cdp, current_filename)
                save_count += 1
      
        # 4. 关闭浏览器
        browser.close()

if __name__ == "__main__":
    run_manual_test()
