import copy
import json
import uuid
from pathlib import Path
# from multiprocessing import Value, Lock

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
default_options: dict = {}

# NORMAL_CUDA = Value('b', True)
# cuda_lock = Lock()
NORMAL_CUDA = True


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
        return (not states.is_running(self.process.state)
                and not states.is_pending(self.process.state)) if self.process else self.__isFinished

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
        self.__isFinished = True
        self.__error = states.is_error(self.process.state)
        self.output_path = self.process.output_vid_path
        self.release()

    def release(self):
        """清除显存"""
        if self.process is not None:
            self.process = None
        if config.RELEASE:
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


@app.get("/health")
async def health():
    # return NORMAL_CUDA.value
    return NORMAL_CUDA


@app.exception_handler(RuntimeError)
async def handle_cuda_error(request, exc):
    # with cuda_lock:
    #     if str(exc).find('CUDA') > -1:
    #         NORMAL_CUDA.value = False
    global NORMAL_CUDA
    if str(exc).find('CUDA') > -1:
        NORMAL_CUDA = False

    raise exc


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
        raise HTTPException(400, detail=f"Failed to parse options as JSON: {str(e)}", )

    if not isinstance(options, dict):
        raise HTTPException(400,
                            detail=f"Unable to parse the 'options' argument into a dictionary; the current format is: {type(options).name}", )

    new_options = copy.deepcopy(default_options)
    new_options.update(options)

    task = Worker(target, source)
    task.set_process(new_options)
    success = submit_worker(task)
    if not success:
        raise HTTPException(500, detail='The task failed to submit.', )

    return JSONResponse(content={
        'process_id': task.process_id
    },
        status_code=status.HTTP_200_OK)


@app.get("/state")
async def state(process_id: str):
    task: Worker = workers.get(process_id)
    if task is None:
        raise HTTPException(404, detail=f'Error process_id: {process_id}', )

    if task.error:
        raise HTTPException(500, detail='The task has been completed, but there might have been some errors, please submit again.', )

    return JSONResponse(content={
        **task.get_state()
    },
        status_code=status.HTTP_200_OK)


@app.get("/download")
async def download(process_id: str):
    task: Worker = workers.get(process_id)
    if task is None:
        raise HTTPException(404, detail=f'Error process_id: {process_id}', )

    if not task.isFinished:
        raise HTTPException(406, detail='The task is not finished yet, please try again later.', )

    if task.error or task.output_path == '' or not os.path.isfile(task.output_path):
        raise HTTPException(500,
                            detail='The task has been completed, but the file was not found. There might have been some errors, please submit again.', )

    return FileResponse(
        path=task.output_path,
        media_type="application/octet-stream",
        filename=Path(task.output_path).name,
    )


def launch():
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
