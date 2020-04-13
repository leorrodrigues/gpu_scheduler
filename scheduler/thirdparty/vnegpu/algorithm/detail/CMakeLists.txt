project (VNEGPU_ALGORITHM_DETAIL)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::vne_algorithm_detail ALIAS ${PROJECT_NAME})
