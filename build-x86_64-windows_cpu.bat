@echo off

set VERSION=v0.10.2
set BAZEL_CONFIG=opt
set PYTHON_BIN_PATH=C:/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe

set OUTPUT_DIR=output
set PACKAGE_DIR=%OUTPUT_DIR%\libmediapipe-%VERSION%-x86_64-windows
set DATA_DIR=%OUTPUT_DIR%\data

if not exist mediapipe (
	git clone https://github.com/google/mediapipe.git
)

pushd mediapipe

git checkout %VERSION%
if not exist mediapipe\c (
	mkdir mediapipe\c
)
copy ..\c\* mediapipe\c

bazel build -c %BAZEL_CONFIG% ^
	--action_env PYTHON_BIN_PATH=%PYTHON_BIN_PATH% ^
	--define MEDIAPIPE_DISABLE_GPU=1 ^
	--verbose_failures ^
	mediapipe/c:mediapipe

	rem --compiler=clang-cl ^

if %ERRORLEVEL% NEQ 0 (
	echo "Build failed"
	exit /b 1
)

popd


if not exist %OUTPUT_DIR% (
	mkdir %OUTPUT_DIR%
	mkdir %PACKAGE_DIR%
	mkdir %PACKAGE_DIR%\include
	mkdir %PACKAGE_DIR%\bin
	mkdir %PACKAGE_DIR%\lib
)

echo "Copying libraries"
copy mediapipe\bazel-bin\mediapipe\c\mediapipe.dll %PACKAGE_DIR%\bin
copy mediapipe\bazel-bin\mediapipe\c\mediapipe.if.lib %PACKAGE_DIR%\lib\mediapipe.lib

echo "Copying header"
copy mediapipe\mediapipe\c\mediapipe.h %PACKAGE_DIR%\include

echo "Copying data"

FOR /D %%G IN (mediapipe\bazel-bin\mediapipe\modules\*) DO (
	echo copying module %%~nG

	if not exist %DATA_DIR%\mediapipe\modules\%%~nG (
		mkdir %DATA_DIR%\mediapipe\modules\%%~nG
	)

	for %%E IN (%%G\*.binarypb) DO (
		copy %%E %DATA_DIR%\mediapipe\modules\%%~nG\%%~nxE
	)

	for %%F IN (%%G\*.tflite) DO (
		copy %%F %DATA_DIR%\mediapipe\modules\%%~nG\%%~nxF
	)
)

copy mediapipe\mediapipe\modules\hand_landmark\handedness.txt %DATA_DIR%\mediapipe\modules\hand_landmark
