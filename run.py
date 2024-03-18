from modules.pipeline.pipeline import build_pipeline
from modules.pipeline.pipeline_options import parse_options
# TODO: Parse args

def main():
    options = parse_options()
    build_pipeline(options)

if __name__ == "__main__":
    main()
