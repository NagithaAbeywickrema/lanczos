cmake_minimum_required(VERSION 3.22)
project(
  rsb
  VERSION 0.0.1
  DESCRIPTION "RSB"
  LANGUAGES C CXX)

option(ENABLE_SERIAL "Build RSB Serial" ON)
option(ENABLE_NOMP "Build RSB Nomp" OFF)
option(ENABLE_CUDA "Build RSB Cuda" OFF)
option(ENABLE_SYCL "Build rsb SYCL" OFF)
option(ENABLE_OPENCL "Build RSB OpenCL" OFF)

if(ENABLE_SERIAL)
  set(SOURCES
      src/main.c
      src/lanczos.h
      src/lanczos-serial.c
      src/print-helper.h
      src/print-helper.c
      src/lanczos-aux.h
      src/lanczos-aux.c
      src/kernels.h
      src/kernels-serial.c)
  add_executable(rsb-serial ${SOURCES})
  target_link_libraries(rsb-serial PRIVATE m)
endif()

if(ENABLE_NOMP)
  set(SOURCES
      src/main.c
      src/lanczos.h
      src/lanczos-nomp.c
      src/lanczos-aux.h
      src/lanczos-aux.c
      src/print-helper.h
      src/print-helper.c
      src/kernels.h
      src/kernels-nomp.c)
  add_executable(rsb-nomp ${SOURCES})
  target_link_libraries(rsb-nomp PRIVATE m)
endif()

if (ENABLE_CUDA)
    enable_language(CUDA)
    set(SOURCES 
    src/main.c
    src/lanczos.h
    src/lanczos-cuda.cu
    src/lanczos-aux.h
    src/lanczos-aux.c
    src/print-helper.h
    src/print-helper.c
    src/kernels.h
    src/kernels-cuda.cu)
    add_executable(rsb-cuda ${SOURCES})
    target_link_libraries(rsb-cuda PRIVATE m)
endif()

if (ENABLE_SYCL)
    set(SOURCES 
    src/main.c 
    src/lanczos-sycl.cpp 
    src/lanczos-aux.c 
    src/print-helper.c
    src/lanczos-aux.h
    src/lanczos.h 
    src/print-helper.h
    src/kernels.h
    src/kernels-sycl.cpp)
    add_executable(rsb-sycl ${SOURCES})
    find_package(IntelDPCPP REQUIRED)
    target_link_libraries(rsb-sycl PRIVATE m -fsycl)
    target_compile_definitions(rsb-sycl PRIVATE ENABLE_SYCL)
endif()

if(ENABLE_OPENCL)
  add_executable(
    rsb_ocl
    src/main.c
    src/lanczos-ocl.c
    src/print-helper.c
    src/lanczos-aux.c
    src/kernels-ocl.c
    src/lanczos.h
    src/kernels.h
    src/print-helper.h)
  find_package(OpenCL REQUIRED)
  target_link_libraries(rsb_ocl PRIVATE OpenCL::OpenCL m)
endif()
