project (AHP LANGUAGES CUDA CXX)

add_subdirectory(hierarchy)

add_library(${PROJECT_NAME} ahp.cpp ahpg.cu)

add_library(sub::ahp ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} cudart)

set_target_properties( ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
    "sub::hierarchy"
)
