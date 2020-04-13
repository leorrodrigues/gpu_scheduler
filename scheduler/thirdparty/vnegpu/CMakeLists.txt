project (VNEGPU)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::vnegpu ALIAS ${PROJECT_NAME})

add_subdirectory(algorithm)
add_subdirectory(generator)
add_subdirectory(util)

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::vnegpu_algorithm"
    INTERFACE "sub::vnegpu_generator"
    INTERFACE "sub::vnegpu_util"
)
