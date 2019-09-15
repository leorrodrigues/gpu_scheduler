project (DATACENTER)

add_subdirectory(tasks)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::datacenter ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::datacenter_tasks"
)
