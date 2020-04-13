project (SPDLOG_SINKS)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::spdlog_sinks ALIAS ${PROJECT_NAME})
