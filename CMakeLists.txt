cmake_minimum_required(VERSION 3.16)
project(NLMeanDemo)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANARD 17)
add_executable(NLMeanDemo nlmtest.cpp nlm.cpp nlm.h)
target_link_libraries(NLMeanDemo ${OpenCV_LIBS})
