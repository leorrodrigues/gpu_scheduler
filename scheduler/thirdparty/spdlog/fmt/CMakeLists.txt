project (SPDLOG_FMT)

add_subdirectory(bundled)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::spdlog_fmt ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::spdlog_bundled"
)
