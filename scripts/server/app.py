import copy
import json
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, status
from fastapi.responses import FileResponse, JSONResponse
import os

from configs import config
from scripts import states
from utils import thread_pool
from scripts.inference import Process

app = FastAPI()
workers = {}
default_options: dict = None


class Worker:

    def __init__(self, video: UploadFile, audio: UploadFile):
        self.process_id = str(uuid.uuid4())
        self.video_path = os.path.join(config.INPUT_DIR, f'{self.process_id[:6]}_{Path(video.filename).name}')
        self.audio_path = os.path.join(config.INPUT_DIR, f'{self.process_id[:6]}_{Path(audio.filename).name}')
        self._save_file(video, self.video_path)
        self._save_file(audio, self.audio_path)
        self.process: Process = None
        self.output_path = ''

        self.__isFinished = False
        self.__state = None
        self.__error = False

    def _save_file(self, file: UploadFile, file_path: str):
        with open(file_path, "wb") as f:
            f.write(file.file.read())

    @property
    def isFinished(self):
        # 为增加state路由的时间容错率，故增加is_pending的判断
        return not states.is_running(self.process.state) and not states.is_pending(self.process.state) if self.process else self.__isFinished

    @property
    def state(self):
        return self.process.state.value if self.process else self.__state

    @property
    def error(self):
        return states.is_error(self.process.state) if self.process else self.__error

    def set_process(self, options: dict):
        self.process = Process(self.video_path, self.audio_path, options=options)
        ...

    def run(self):
        print(f'开始执行任务：{self.process_id}')
        self.process.run()
        print(f'任务结束：{self.process_id}')
        self.__state = self.process.state.value
        self.__isFinished = not states.is_running(self.process.state)
        self.__error = states.is_error(self.process.state)
        self.output_path = self.process.output_vid_path
        self.release()

    def release(self):
        """清除显存"""
        if self.process is not None:
            self.process = None
        torch.cuda.empty_cache()

    def get_state(self):
        return {
            'finished': self.isFinished,
            'state': self.state,
        }


def submit_worker(worker: Worker) -> bool:
    print(f'提交任务：{worker.process_id}')
    success = thread_pool.submit(worker.run)
    if success:
        workers[worker.process_id] = worker
    else:
        worker.release()
    return success


@app.post("/submit")
async def submit(
        options: str = Form(None),
        source: UploadFile = File(...),
        target: UploadFile = File(...),
):
    try:
        options = options or '{}'
        options = json.loads(options)
    except Exception as e:
        return JSONResponse(content={"message": f'无法解析options为JSON：{str(e)}'},
                            status_code=status.BAD_REQUEST)

    if not isinstance(options, dict):
        return JSONResponse(content={"message": f'无法解析options参数为dict，当前格式为：{type(options).__name__}'},
                            status_code=status.BAD_REQUEST)

    new_options = copy.deepcopy(default_options)
    new_options.update(options)

    worker = Worker(target, source)
    worker.set_process(new_options)
    success = submit_worker(worker)
    if success:
        return JSONResponse(content={
            'code': 200,
            'process_id': worker.process_id
        },
            status_code=status.HTTP_200_OK)
    else:
        return JSONResponse(content={
            'code': 200,
            'msg': '任务提交失败'
        },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/state")
async def state(process_id: str):
    worker: Worker = workers.get(process_id)
    if worker is None:
        return JSONResponse(content={
            'code': 200,
            'msg': f'Error process_id: {process_id}'
        },
            status_code=status.HTTP_404_NOT_FOUND)

    return JSONResponse(content={
        'code': 200 if not worker.error else 406,
        **worker.get_state()
    },
        status_code=status.HTTP_200_OK)


@app.get("/download")
async def download(process_id: str):
    worker: Worker = workers.get(process_id)
    if worker is None:
        return JSONResponse(content={
            'code': 200,
            'msg': f'Error process_id: {process_id}'
        },
            status_code=status.HTTP_404_NOT_FOUND)

    if not worker.isFinished:
        return JSONResponse(content={
            'code': 200,
            'msg': f'The work is not finished yet, please try again later.'
        },
            status_code=status.HTTP_406_NOT_ACCEPTABLE)

    if worker.error or worker.output_path == '' or not os.path.isfile(worker.output_path):
        return HTTPException(404,
                             detail='The work has been completed, but the file was not found. There might have been some errors, please submit again.', )

    return FileResponse(
        path=worker.output_path,
        media_type="application/octet-stream",  # 可以根据文件类型进行调整
        headers={"Content-Disposition": f"attachment; filename={Path(worker.output_path).name}"}
    )


def launch():
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
