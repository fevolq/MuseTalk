from argparse import ArgumentParser

from configs import config
from scripts import download


def cli():
    parser = ArgumentParser()

    # generate
    parser.add_argument("--api", action="store_true", help="run the program with api")
    parser.add_argument('--port', type=int, default=config.PORT, help="run the program with api")
    parser.add_argument("--max-pool", type=int, default=config.MAX_POOL, help="Max pool size (must be >= 1)")
    parser.add_argument("--float16", action="store_true", help="Whether use float16 to speed up inference", )

    # options
    group_options = parser.add_argument_group('options')
    group_options.add_argument("--bbox_shift", type=int, default=0)
    group_options.add_argument("--fps", type=int, default=25)
    group_options.add_argument("--batch_size", type=int, default=8)
    group_options.add_argument("--use_saved_coord", action="store_true", help='use saved coordinate to save time')

    run(parser)


def apply_parser(args) -> None:
    config.PORT = args.port
    config.MAX_POOL = args.max_pool
    config.FLOAT16 = args.float16


def run(program: ArgumentParser) -> None:
    download.download_models()

    args = program.parse_args()
    apply_parser(args)
    default_options = {
        'bbox_shift': args.bbox_shift,
        'fps': args.fps,
        'batch_size': args.batch_size,
        'use_saved_coord': args.use_saved_coord,
    }

    if args.api:
        print(f'------ Start with api ------')
        from scripts.server import app
        app.default_options = default_options
        app.launch()
    else:
        from scripts.server import gui
        print(f'------ Start with gui ------')
        gui.launch()
