from enum import unique, Enum


@unique
class State(Enum):
    PENDING = '空闲中'
    END = '已结束'
    ERROR = '错误'

    PREPROCESS_INPUT_IMG = '准备图片帧'
    CROP_IMG = '截取帧区域'
    INFERENCE = '推理'
    PAD_IMG = '生成图片帧'
    GEN_VIDEO = '合成视频'


def is_pending(state):
    return state == State.PENDING


def is_error(state):
    return state == State.ERROR


def is_running(state):
    return state not in (
        State.PENDING,
        State.END,
        State.ERROR,
    )
