project (MULTICRITERIA)

add_subdirectory(topsis)
add_subdirectory(ahp)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::rank_multicriteria ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::ahp"
    INTERFACE "sub::topsis"
)
