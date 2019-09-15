project (SPDLOG_DETAILS)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::spdlog_details ALIAS ${PROJECT_NAME})
