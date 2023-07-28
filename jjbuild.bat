@echo off

set psshell=C:\Program Files\Git\bin\sh.exe

set BAZEL_CONFIG=opt
set PYTHON_BIN_PATH=C:/Users/Administrator/AppData/Local/Programs/Python/Python310/python.exe
set GLOG_logtostderr=1

pushd mediapipe

rem bazel build -c %BAZEL_CONFIG% --action_env PYTHON_BIN_PATH=%PYTHON_BIN_PATH% --define MEDIAPIPE_DISABLE_GPU=1 --verbose_failures  mediapipe/examples/desktop/hello_world:hello_world
rem "bazel-bin/mediapipe/examples/desktop/hello_world/hello_world.exe"

rem bazel build -c %BAZEL_CONFIG% --action_env PYTHON_BIN_PATH=%PYTHON_BIN_PATH% --define MEDIAPIPE_DISABLE_GPU=1 --verbose_failures  mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu

bazel build -c %BAZEL_CONFIG% --action_env PYTHON_BIN_PATH=%PYTHON_BIN_PATH% --define MEDIAPIPE_DISABLE_GPU=1 --verbose_failures  mediapipe/tasks:internal

popd