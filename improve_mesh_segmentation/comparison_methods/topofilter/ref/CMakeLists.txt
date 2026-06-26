cmake_minimum_required(VERSION 3.15)
project(PythonGraphPers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w -fPIC")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(${PROJECT_NAME}
  pythonGraphTopoFix_withCompInfo.cpp
)

install(
  TARGETS
    ${PROJECT_NAME}
  DESTINATION
    .
)
