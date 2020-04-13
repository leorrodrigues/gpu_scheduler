project (VNE_ALGORITHM_SERIAL_DETAIL)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::vne_algorithm_serial_detail ALIAS ${PROJECT_NAME})
