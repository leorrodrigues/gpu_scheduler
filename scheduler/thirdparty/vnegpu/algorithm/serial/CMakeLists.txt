project (VNE_ALGORITHM_SERIAL)

add_subdirectory(detail)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::vne_algorithm_serial ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::vne_algorithm_serial_detail"
)
