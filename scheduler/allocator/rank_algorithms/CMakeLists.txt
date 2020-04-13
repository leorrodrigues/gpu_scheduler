project (RANK LANGUAGES CUDA CXX)

add_subdirectory(multicriteria)
add_subdirectory(standard)
add_subdirectory(user_defined)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::rank ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::rank_multicriteria"
    INTERFACE "sub::rank_standard"
    INTERFACE "sub::user_defined_rank"
)
