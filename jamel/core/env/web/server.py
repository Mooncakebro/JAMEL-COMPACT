# pip install aiofiles fastapi uvicorn
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import sys

app = FastAPI()

# 假设你的网站根目录是这个路径，你可以根据实际情况修改
# html=True 非常关键！它让 FastAPI 访问目录时自动寻找 index.html
app.mount("/", StaticFiles(directory=sys.argv[1], html=True), name="static")

if __name__ == "__main__":
    # workers=4 开启 4 个工作进程，轻松应对高并发
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=20)