project (ALLOCATOR LANGUAGES CUDA CXX)

add_subdirectory(links)
add_subdirectory(rank_algorithms)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::allocator ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
    INTERFACE "sub::allocator_links"
    INTERFACE "sub::rank"
)
