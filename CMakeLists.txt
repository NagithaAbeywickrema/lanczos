cmake_minimum_required(VERSION 3.22)
project(rsb VERSION 0.0.1 DESCRIPTION "RSB" LANGUAGES C)

option(ENABLE_NOMP "Build rsb serial" ON)

if (ENABLE_NOMP)
    set(SOURCES src/main.c src/lanczos.c src/lanczos.h)
    add_executable(rsb_nomp ${SOURCES})
    target_link_libraries(rsb_nomp PRIVATE m)
endif()
