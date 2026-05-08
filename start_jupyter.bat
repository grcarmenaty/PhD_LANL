@echo off
set D=C:\Users\user\Documents\Claude\Projects\PhD\3SBB_data
"%D%\.venv\Scripts\jupyter.exe" lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir="%D%"
pause