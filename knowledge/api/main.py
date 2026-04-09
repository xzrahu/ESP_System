# 创建fastapi实例 管理所有的路由
import logging

import uvicorn
from fastapi import FastAPI

from api.routers import router

def create_app() -> FastAPI:
    # 1 创建FastApi
    app = FastAPI(title="Knowledge API")

    # 2 注册各种路由
    app.include_router(router)


    return app

if __name__ == '__main__':
    print("1.启动web服务器")
    try:
        uvicorn.run(create_app(),host="127.0.0.1",port=8001)
        logging.info("2.服务器启动成功")
    except Exception as e:
        logging.error(f"2.服务器启动失败 原因{str(e)}")
