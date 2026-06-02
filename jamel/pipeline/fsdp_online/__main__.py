from .main import FSDPOnlinePolicyGradientPipeline


if __name__ == "__main__":
    pipeline = FSDPOnlinePolicyGradientPipeline()
    try:
        pipeline.run()
    finally:
        pipeline.close()
