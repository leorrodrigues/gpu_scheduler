project (CLUSTERING LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} mclInterface.cu)
add_library(sub::clustering ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} cudart)

set_target_properties( ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
