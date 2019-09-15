project (VNEGPU_ALGORITHM_GENERATOR)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::vnegpu_algorithm_generator ALIAS ${PROJECT_NAME})
