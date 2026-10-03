
cmake_minimum_required(VERSION 3.20)
project(IterativeSolvers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(itsolvers INTERFACE)
target_include_directories(itsolvers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(itsolvers INTERFACE Eigen3::Eigen)

add_executable(gram_schmidt_stability examples/gram_schmidt_stability.cpp)
target_link_libraries(gram_schmidt_stability PRIVATE itsolvers)
target_compile_options(gram_schmidt_stability PRIVATE -Wall -Wextra -Wpedantic)

add_executable(jacobi_gs_demo examples/jacobi_gs_demo.cpp)
target_link_libraries(jacobi_gs_demo PRIVATE itsolvers)
target_compile_options(jacobi_gs_demo PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(test_solvers tests/test_solvers.cpp)
target_link_libraries(test_solvers PRIVATE itsolvers)
target_compile_options(test_solvers PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME solvers COMMAND test_solvers)

add_executable(test_arnoldi tests/test_arnoldi.cpp)
target_link_libraries(test_arnoldi PRIVATE itsolvers)
target_compile_options(test_arnoldi PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME arnoldi COMMAND test_arnoldi)