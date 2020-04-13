project (VNEGPU_UTIL)

add_subdirectory(confiabilidade)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::vnegpu_util ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::vnegpu_util_confiabilidade"
)
