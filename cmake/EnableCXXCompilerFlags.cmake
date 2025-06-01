# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include(CheckCXXCompilerFlag)

# For easier adding of CXX compiler flags
function(seal_enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
        message(STATUS "Adding CXX compiler flag: ${flag} ...")
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()

if(NOT MSVC)
    seal_enable_cxx_compiler_flag_if_supported("-Wall")
    seal_enable_cxx_compiler_flag_if_supported("-Wextra")
    seal_enable_cxx_compiler_flag_if_supported("-Wconversion")
    seal_enable_cxx_compiler_flag_if_supported("-Wshadow")
    seal_enable_cxx_compiler_flag_if_supported("-pedantic")
    seal_enable_cxx_compiler_flag_if_supported("-O3")
    seal_enable_cxx_compiler_flag_if_supported("-march=rv64gcv")
    seal_enable_cxx_compiler_flag_if_supported("-mabi=lp64d")
    seal_enable_cxx_compiler_flag_if_supported("-ftree-vectorize")
    seal_enable_cxx_compiler_flag_if_supported("-fopenmp")
    seal_enable_cxx_compiler_flag_if_supported("-fopt-info-vec-optimized")
endif()
