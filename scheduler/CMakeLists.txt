cmake_minimum_required(VERSION 3.15.3 FATAL_ERROR)
project (gpu_scheduler)

enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -use_fast_math -lineinfo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(GCC_FLAGS "-std=c++17 -Wall -D_GLIBCXX_ASSERTIONS -D_FORTIFY_SOURCE=2 -fasynchronous-unwind-tables -fstack-protector-strong  -pipe -Werror=format-security  -Wduplicated-branches  -Wlogical-op -Wformat=2 -Wduplicated-cond -fconcepts -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_FLAGS}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
#Add sub directories
add_subdirectory(thirdparty)
add_subdirectory(main_resources)
add_subdirectory(objective_functions)
add_subdirectory(topology)
add_subdirectory(clustering)
add_subdirectory(multicriteria)
add_subdirectory(datacenter)
add_subdirectory(allocator)

add_executable(${PROJECT_NAME}.out json.cpp builder.cu gpu_scheduler.cu)

target_link_libraries(${PROJECT_NAME}.out
    -lcublas
    -lcusparse
    -lboost_program_options
    "sub::spdlog"
    "sub::main_resources"
    "sub::obj_func"
    "sub::topology"
    "sub::clustering"
    "sub::topsis"
    "sub::ahp"
    "sub::datacenter_tasks"
    "sub::allocator"
)

file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ahp)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/topsis)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/datacenter)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/datacenter/fat_tree)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/requests)

configure_file(main_resources/resourcesData.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
configure_file(main_resources/resourcesSchema.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
configure_file(datacenter/json/hostsSchema.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/datacenter COPYONLY)

message("-- Copying the json files")

file(GLOB AHP_JSON
  "multicriteria/ahp/json/*.json"
)
file(COPY ${AHP_JSON} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ahp)

file(GLOB TOPSIS_JSON
  "multicriteria/topsis/json/*.json"
)
file(COPY ${TOPSIS_JSON} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/topsis)

file(GLOB DATACENTER_JSON
  "datacenter/json/fat_tree"
)
file(COPY ${DATACENTER_JSON} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/datacenter/)

file(GLOB REQUESTS
  "requests"
)
file(COPY ${REQUESTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

message("-- Creating the logs directories")
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/logs)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/logs/test1)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/logs/test2)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/logs/test3)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/logs/test4)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/logs/test5)
