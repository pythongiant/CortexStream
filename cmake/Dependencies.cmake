include(FetchContent)

# Set minimum CMake policy version to handle old subprojects
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# tokenizers-cpp: C++ bindings for HuggingFace tokenizers
# This library provides native C++ support for loading tokenizer.json files
FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)

# Set build options for tokenizers-cpp
set(TOKENIZERS_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(tokenizers_cpp)
