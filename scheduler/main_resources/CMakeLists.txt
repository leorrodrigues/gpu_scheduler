project (MAIN_RESOURCES)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::main_resources ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::spdlog"
)
