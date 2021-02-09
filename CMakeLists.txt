cmake_minimum_required (VERSION 2.6)
project (sample)

find_package(PkgConfig)
pkg_check_modules(PC_DBUS REQUIRED dbus-1)
pkg_check_modules(PC_GLIB REQUIRED glib-2.0)
pkg_check_modules(PC_GIO REQUIRED gio-2.0)
pkg_check_modules(PC_GIO_UNIX REQUIRED gio-unix-2.0)
pkg_check_modules(PC_MYSQLCLIENT REQUIRED mysqlclient)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

set (sample_VERSION_MAJOR 1)
set (sample_VERSION_MINOR 0)
set(CMAKE_VERBOSE_MAKEFILE ON)
option(BUILD_C_MODULE_API "c_module_api" OFF)
option(BUILD_CUDA_MODULE_API "cuda_module_api" ON)
option(BUILD_FOR_NVIDIA_VISUAL_PROFILER "build for nvidia profiler" OFF)
execute_process( COMMAND sudo mkdir -p /usr/local/cuda/bin/ )
execute_process( COMMAND sudo ln -sf /usr/bin/gcc-7 /usr/local/cuda/bin/gcc )
execute_process( COMMAND sudo ln -sf /usr/bin/g++-7 /usr/local/cuda/bin/g++ )
execute_process( COMMAND sudo ln -sf /usr/bin/gcc-7 /usr/bin/gcc )
execute_process( COMMAND sudo ln -sf /usr/bin/g++-7 /usr/bin/g++ )
set(CMAKE_C_COMPILER /usr/local/cuda/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/local/cuda/bin/g++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")

configure_file (
  "${PROJECT_SOURCE_DIR}/sample.h.in"
  "${PROJECT_BINARY_DIR}/sample.h"
  )

MESSAGE( STATUS "PROJECT_BINARY_DIR: " ${PROJECT_BINARY_DIR} " PROJECT_SOURCE_DIR " ${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/includes/ ${PC_PTHREAD_INCLUDE_DIRS} ${PC_DBUS_INCLUDE_DIRS} ${PC_GLIB_INCLUDE_DIRS})

if (BUILD_C_MODULE_API)
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/C_MODULE/)
endif (BUILD_C_MODULE_API)

if (BUILD_CUDA_MODULE_API)
 	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/CUDA_MODULE/)
endif (BUILD_CUDA_MODULE_API)

install (FILES ${PROJECT_BINARY_DIR}/sample.h DESTINATION include)

if (BUILD_C_MODULE_API)
	add_subdirectory (C_MODULE)
endif (BUILD_C_MODULE_API)

if (BUILD_CUDA_MODULE_API)
	add_subdirectory (CUDA_MODULE)
endif (BUILD_CUDA_MODULE_API)
#SET(CMAKE_C_LINK_EXECUTABLE nvcc)
add_executable(sample src/common.c)
target_link_libraries(sample  ${PC_GLIB_LIBRARY_DIRS} ${PC_GIO_LIBRARY_DIRS} ${PC_PTHREAD_LIBRARY_DIRS} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY} ${CUDA_STUBS_LIBS})

if (BUILD_C_MODULE_API)
	target_link_libraries(sample c_module_api)
endif (BUILD_C_MODULE_API)
if (BUILD_CUDA_MODULE_API)
	target_link_libraries(sample cuda_module_api)
endif (BUILD_CUDA_MODULE_API)

install (TARGETS sample DESTINATION bin)
install (FILES ${PROJECT_BINARY_DIR}/sample.h DESTINATION include)
