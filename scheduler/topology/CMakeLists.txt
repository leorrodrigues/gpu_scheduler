project (TOPOLOGY)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::topology ALIAS ${PROJECT_NAME})
