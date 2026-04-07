@echo off
setlocal
echo ========================================================
echo  MASC-CSC | LLM Verifier LoRA 微调 (Windows 本地测试)
echo ========================================================
echo.

:: 🚨 请在这里填写你下载的 Qwen-7B 文件夹的绝对路径！
:: 比如: set MODEL_PATH=E:\pretrained\Qwen1.5-7B
set MODEL_PATH=.\pretrained\Qwen-7B

set DATA_PATH=.\datasets\mock_llm_data.jsonl
set OUTPUT_DIR=.\ckpt\llm_lora_qwen

if not exist "%MODEL_PATH%" (
    echo [ERROR] 找不到基座模型目录 "%MODEL_PATH%"
    echo 请右键编辑 scripts\train_llm_lora.bat，修改第 8 行的 MODEL_PATH 变量！
    exit /b 1
)

if not exist "%DATA_PATH%" (
    echo [ERROR] 找不到微调训练数据 "%DATA_PATH%"
    exit /b 1
)

echo [INFO] 模型路径: %MODEL_PATH%
echo [INFO] 开始启动 4-bit 量化 LoRA 训练...
echo.

python scripts\train_llm_lora.py ^
    --model_name_or_path "%MODEL_PATH%" ^
    --data_path "%DATA_PATH%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --use_4bit ^
    --per_device_train_batch_size 1 ^
    --gradient_accumulation_steps 4 ^
    --learning_rate 2e-4 ^
    --num_train_epochs 3 ^
    --lora_r 16 ^
    --lora_alpha 32 ^
    --logging_steps 1

if errorlevel 1 (
    echo.
    echo [ERROR] 运行期间发生错误！
    exit /b 1
)

echo.
echo [SUCCESS] LoRA 微调顺利完成！权重输出在 %OUTPUT_DIR%
exit /b 0
