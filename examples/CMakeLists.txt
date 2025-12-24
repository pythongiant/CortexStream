# Example applications for CortexStream

# tokenizers_cpp is already fetched in Dependencies.cmake
option(WITH_TOKENIZERS_CPP "Enable token decoding in examples" ON)

# Simple inference example
add_executable(simple_inference simple_inference.cpp)
target_link_libraries(simple_inference PRIVATE cortexstream)
target_include_directories(simple_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Multi-request server example
add_executable(multi_request_server multi_request_server.cpp)
target_link_libraries(multi_request_server PRIVATE cortexstream)
target_include_directories(multi_request_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Benchmark example
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE cortexstream)
target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# HuggingFace inference example
add_executable(huggingface_inference huggingface_inference.cpp)
target_link_libraries(huggingface_inference PRIVATE cortexstream)
target_include_directories(huggingface_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(WITH_TOKENIZERS_CPP)
    target_link_libraries(huggingface_inference PRIVATE tokenizers_cpp)
    target_include_directories(huggingface_inference PRIVATE ${tokenizers_cpp_SOURCE_DIR}/include)
    target_compile_definitions(huggingface_inference PRIVATE CORTEXSTREAM_WITH_TOKENIZERS_CPP=1)
endif()

# Optimization flags for examples
foreach(example
    simple_inference
    multi_request_server
    benchmark
    huggingface_inference
)
    target_compile_options(${example} PRIVATE
        -Wall -Wextra -Wpedantic
        -O3 -march=native
    )
    
    # Apple Silicon specific optimizations
    if(APPLE)
        target_compile_options(${example} PRIVATE
            -mtune=apple-m1
            -ffast-math
        )
    endif()
endforeach()
