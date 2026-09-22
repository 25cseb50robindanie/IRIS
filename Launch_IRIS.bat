@echo off
setlocal
title IRIS - Satellite Intelligence Desktop
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1"
