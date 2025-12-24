# Source files for CortexStream library

set(CORTEXSTREAM_SOURCES
    cache/kv_cache.cpp
    cache/allocator.cpp
    engine/engine.cpp
    engine/inference_engine.cpp
    engine/scheduler.cpp
    model/model_backend.cpp
    model/sampling.cpp
    model/tokenizer.cpp
    request/request.cpp
    request/response.cpp
    response/response.cpp
    utils/log.cpp
    utils/metrics.cpp
)

# Create CortexStream library
add_library(cortexstream ${CORTEXSTREAM_SOURCES})

# Set target include directories
target_include_directories(cortexstream PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Link required libraries
target_link_libraries(cortexstream PUBLIC
    pthread
    m
)

# MLX framework integration (Apple Silicon GPU acceleration)
find_package(mlx QUIET)
if(mlx_FOUND)
    target_link_libraries(cortexstream PUBLIC mlx)
    target_compile_definitions(cortexstream PUBLIC MLX_AVAILABLE=1)
    message(STATUS "MLX framework found - GPU acceleration enabled")
else()
    message(STATUS "MLX framework not found - CPU fallback only")
endif()

# OpenMP for parallelization
find_package(OpenMP QUIET)
if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
    target_link_libraries(cortexstream PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - parallel processing enabled")
else()
    message(STATUS "OpenMP not found - install with: brew install libomp")
    # Try to link with system OpenMP if available
    if(APPLE)
        # Try Homebrew libomp location
        if(EXISTS "/opt/homebrew/opt/libomp")
            target_include_directories(cortexstream PUBLIC /opt/homebrew/opt/libomp/include)
            target_link_directories(cortexstream PUBLIC /opt/homebrew/opt/libomp/lib)
            target_link_libraries(cortexstream PUBLIC omp)
            message(STATUS "Found Homebrew libomp - linking manually")
        endif()
    endif()
endif()

# JSON library for HuggingFace model config parsing
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(cortexstream PUBLIC nlohmann_json::nlohmann_json)
    message(STATUS "nlohmann_json found - HuggingFace support enabled")
else()
    message(STATUS "nlohmann_json not found - HuggingFace support disabled")
endif()

# CURL for downloading HuggingFace models
find_package(CURL QUIET)
if(CURL_FOUND)
    target_link_libraries(cortexstream PUBLIC CURL::libcurl)
    target_compile_definitions(cortexstream PUBLIC CURL_AVAILABLE=1)
    message(STATUS "CURL found - model download support enabled")
else()
    message(STATUS "CURL not found - model download support disabled")
endif()

# Enable all warnings and optimization
target_compile_options(cortexstream PRIVATE
    -Wall -Wextra -Wpedantic
    -O3 -march=native
)

# Apple Silicon specific optimizations
if(APPLE)
    target_compile_options(cortexstream PRIVATE
        -mtune=apple-m1
        -ffast-math
    )
endif()
