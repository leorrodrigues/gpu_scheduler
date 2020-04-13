project (USER_DEFINED_RANK LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} INTERFACE)
add_library(sub::user_defined_rank ALIAS ${PROJECT_NAME})
