import subprocess

# 定义要运行的 Python 脚本及其参数（字符串形式）
commands = [
    # r'python compress.py --model_path D:\code\GS\gaussian-splatting\output\350-1600-w_b\1 --data_device "cuda" --output_vq D:\code\GS\c3dgs\output\w-b\1 --use_image_w',
    # r'python compress.py --model_path D:\code\GS\gaussian-splatting\output\350-1600-w_b\2 --data_device "cuda" --output_vq D:\code\GS\c3dgs\output\w-b\2 --use_image_w',
    # r'python compress.py --model_path D:\code\GS\gaussian-splatting\output\350-1600-w_b\3 --data_device "cuda" --output_vq D:\code\GS\c3dgs\output\w-b\3 --use_image_w',
    # r'python compress.py --model_path D:\code\GS\gaussian-splatting\output\350-1600-w_b\4 --data_device "cuda" --output_vq D:\code\GS\c3dgs\output\w-b\4 --use_image_w',
    # r'python compress.py --model_path D:\code\GS\gaussian-splatting\output\350-1600-w_b\5 --data_device "cuda" --output_vq D:\code\GS\c3dgs\output\w-b\5 --use_image_w',
    # r'python compress.py --model_path D:\code\GS\gaussian-splatting\output\350-1600-w_b\6 --data_device "cuda" --output_vq D:\code\GS\c3dgs\output\w-b\6 --use_image_w',
    # r'python compress.py --model_path D:\code\GS\gaussian-splatting\output\350-1600-w_b\7 --data_device "cuda" --output_vq D:\code\GS\c3dgs\output\w-b\7 --use_image_w',
    # r'python compress.py --model_path D:\code\GS\gaussian-splatting\output\350-1600-w_b\8 --data_device "cuda" --output_vq D:\code\GS\c3dgs\output\w-b\8 --use_image_w',
    # r'python compress.py --model_path D:\code\GS\gaussian-splatting\output\350-1600-w_b\mergeRes --data_device "cuda" --output_vq D:\code\GS\c3dgs\output\mergeRes --use_image_w',

    r'python render.py --skip_test -r 1 --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\1',
    r'python render.py --skip_test -r 1 --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\2',
    r'python render.py --skip_test -r 1 --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\3',
    r'python render.py --skip_test -r 1 --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\4',
    r'python render.py --skip_test -r 1 --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\5',
    r'python render.py --skip_test -r 1 --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\6',
    r'python render.py --skip_test -r 1 --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\7',
    r'python render.py --skip_test -r 1 --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\8',
    r'python render.py --skip_test -r 1 --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\mergeRes',

    r'python metrics.py --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\1',
    r'python metrics.py --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\2',
    r'python metrics.py --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\3',
    r'python metrics.py --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\4',
    r'python metrics.py --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\5',
    r'python metrics.py --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\6',
    r'python metrics.py --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\7',
    r'python metrics.py --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\8',
    r'python metrics.py --calculate_image_w -m D:\code\GS\c3dgs\output\w-b\mergeRes',
]

def run_commands(command_list):
    for command in command_list:
        try:
            print(f"Running command: {command}")
            # 使用 subprocess.run 运行命令
            subprocess.run(command, shell=True, check=True)
            print(f"Command completed successfully: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {command}")
            print(f"Error message: {e}")
            break  # 如果某个命令失败，停止执行后续命令

if __name__ == "__main__":
    run_commands(commands)