@echo off
rem Current entry point. The frozen v3 runner is run_assembly_context_v3_legacy.bat.
call "%~dp0run_adaptive_p10_comparison.bat" %*
exit /b %ERRORLEVEL%
