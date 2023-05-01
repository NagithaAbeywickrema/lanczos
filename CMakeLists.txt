cmake_minimum_required(VERSION 3.22)
project(
  rsb
  VERSION 0.0.1
  DESCRIPTION "RSB"
  LANGUAGES C)

option(ENABLE_SERIAL "Build rsb serial" ON)
option(ENABLE_NOMP "Build rsb serial" ON)

if(ENABLE_SERIAL)
  set(SOURCES src/main.c src/lanczos-serial.c src/print-helper.c src/lanczos.h)
  add_executable(rsb_serial ${SOURCES})
  target_link_libraries(rsb_serial PRIVATE m)
endif()

if(ENABLE_NOMP)
  set(SOURCES src/main.c src/lanczos-nomp.c src/print-helper.c src/mtx.c
              src/lanczos.h src/mtx.h src/print-helper.h)
  add_executable(rsb_nomp ${SOURCES})
  target_link_libraries(rsb_nomp PRIVATE m)
endif()
