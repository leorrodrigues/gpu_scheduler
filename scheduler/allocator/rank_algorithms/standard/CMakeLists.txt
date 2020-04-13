project (RANK_STANDARD LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::rank_standard ALIAS ${PROJECT_NAME})
