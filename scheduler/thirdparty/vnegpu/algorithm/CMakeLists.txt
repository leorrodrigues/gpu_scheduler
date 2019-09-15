project (VNEGPU_ALGORITHM)

add_subdirectory(detail)
add_subdirectory(serial)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::vnegpu_algorithm ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::vnegpu_algorithm_detail"
    INTERFACE "sub::vnegpu_algorithm_serial"
)
