project (SPDLOG)

add_subdirectory(details)
add_subdirectory(fmt)
add_subdirectory(sinks)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::spdlog ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::spdlog_details"
    INTERFACE "sub::spdlog_fmt"
    INTERFACE "sub::spdlog_sinks"
)
