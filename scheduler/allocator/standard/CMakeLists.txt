project (ALLOCATOR_STANDARD LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::allocator_standard ALIAS ${PROJECT_NAME})
