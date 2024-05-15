基于[开源项目](https://github.com/TMElyralab/MuseTalk)

1. api服务，基于线程池实现异步调用。
     ```shell
     python run.py --api
      ```
2. Docker
   1. .env配置文件 `cp .env.template .env`
   2. 启动
      ```shell
      # 构建镜像
      docker compose build

      # 启动
      docker compose up -d
	  ```