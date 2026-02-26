@echo off
setlocal
cd /d "%~dp0"

echo [1/3] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python não encontrado no path.
    echo Instale o Python 3.10+ e tente novamente.
    pause
    exit /b 1
)

echo [2/3] Instalando dependencias...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Erro ao instalar dependencias.
    pause
    exit /b 1
)

echo [3/3] Instalação concluída com sucesso!
echo Para executar: python main.py
pause
