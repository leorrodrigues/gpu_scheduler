project (ALLOCATOR_LINKS LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::allocator_links ALIAS ${PROJECT_NAME})
