import paddleocr
print(f"PaddleOCR version: {paddleocr.__version__}")

# If you use the local paddle_static inference engine, you can further verify PaddlePaddle and GPU availability
import paddle
print(f"Paddle version: {paddle.__version__}")
print(f"GPU available: {paddle.is_compiled_with_cuda()}")
print(f"GPU count: {paddle.device.cuda.device_count()}")