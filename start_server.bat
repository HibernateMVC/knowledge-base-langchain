@echo off
REM 知识库系统启动脚本
setlocal

REM 设置环境变量
echo 正在设置环境变量...
set DASHSCOPE_API_KEY=%1

REM 验证参数
if "%DASHSCOPE_API_KEY%"=="" (
    echo 错误: 请输入通义千问API密钥
    echo 使用方法: start_server.bat ^<DASHSCOPE_API_KEY^>
    pause
    exit /b 1
)

REM 创建必要目录
echo 创建必要目录...
if not exist "data\uploads" mkdir data\uploads
if not exist "logs" mkdir logs

REM 启动服务
echo 正在启动知识库系统...
python -m src.backend.main

pause