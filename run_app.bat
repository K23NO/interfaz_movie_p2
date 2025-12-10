@echo off
setlocal

REM Cambia al directorio del proyecto
cd /d "%~dp0"

REM Asegura dependencias necesarias para Parquet y Streamlit
python -m pip install --quiet --upgrade pip
python -m pip install --quiet streamlit pyarrow fastparquet

REM Ejecuta la aplicación de Streamlit
streamlit run app.py

REM Mantiene la ventana abierta para ver cualquier mensaje
pause
