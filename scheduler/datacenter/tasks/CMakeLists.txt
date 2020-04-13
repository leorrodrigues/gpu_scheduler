project (DATACENTER_TASKS)

add_library(${PROJECT_NAME}
    task.cpp
    pod.cpp
    container.cpp
)

add_library(sub::datacenter_tasks ALIAS ${PROJECT_NAME})
