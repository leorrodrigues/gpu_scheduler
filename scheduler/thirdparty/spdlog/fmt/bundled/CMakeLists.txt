project (SPDLOG_BUNDLED)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::spdlog_bundled ALIAS ${PROJECT_NAME})
