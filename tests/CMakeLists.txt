# Test suite for CortexStream

# Enable testing
enable_testing()

# Find GTest
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GTest not found - building tests with basic framework")
    
    # Define test executables without GTest
    set(TEST_SOURCES
        test_engine.cpp
        test_kvcache.cpp
        test_scheduler.cpp
    )
    
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE cortexstream)
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
else()
    message(STATUS "GTest found - using GTest framework")
    
    set(TEST_SOURCES
        test_engine.cpp
        test_kvcache.cpp
        test_scheduler.cpp
    )
    
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE
            cortexstream
            GTest::GTest
            GTest::Main
        )
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Add custom test target
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${TEST_TARGETS}
)
