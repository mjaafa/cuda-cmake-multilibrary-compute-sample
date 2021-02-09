# DIRTY CMAKEFILE		      

SET(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> arv <TARGET> <LINK_FLAGS> <OBJECTS>" "<CMAKE_RANLIB> <TARGET> ")		      
SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler -fpic)
include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${PC_PTHREAD_INCLUDE_DIRS} ${CUDA_CUDART_LIBRARY})
cuda_add_library(cuda_module_api SHARED cuda_module.c cuda_module.cu)							    
install (TARGETS cuda_module_api DESTINATION lib)
#install (FILES matrixMul_kernel64.ptx DESTINATION bin)
