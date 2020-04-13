project (OBJECTIVE_FUNCTION)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::obj_func ALIAS ${PROJECT_NAME})
