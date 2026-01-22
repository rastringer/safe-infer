cmake_minimum_required(VERSION 3.20)
project(safe_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(safe_infer 
	src/tensor_shape.cpp
	src/tensor.cpp
	src/planner.cpp
	src/executor.cpp
)

target_include_directories(safe_infer PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Warnings 
if (MSVC)
	target_compile_options(safe_infer PRIVATE /W4)
else()
	target_compile_options(safe_infer PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

add_executable(test_tensor_shape tests/test_tensor_shape.cpp)
target_link_libraries(test_tensor_shape PRIVATE safe_infer)
add_test(NAME test_tensor_shape COMMAND test_tensor_shape)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE safe_infer)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_planner tests/test_planner.cpp)
target_link_libraries(test_planner PRIVATE safe_infer)
add_test(NAME test_planner COMMAND test_planner)

add_executable(example_simple_relu examples/simple_relu.cpp)
target_link_libraries(example_simple_relu PRIVATE safe_infer)

# Examples (built, not tested)
add_executable(example_shape_mismatch examples/attacks/shape_mismatch.cpp)
target_link_libraries(example_shape_mismatch PRIVATE safe_infer)

add_executable(example_missing_input examples/attacks/missing_input.cpp)
target_link_libraries(example_missing_input PRIVATE safe_infer)

add_executable(test_executor tests/test_executor.cpp)
target_link_libraries(test_executor PRIVATE safe_infer)
add_test(NAME test_executor COMMAND test_executor)
