project (HIERARCHY)

add_library(${PROJECT_NAME}
    hierarchy_resource.cpp
    edge.cpp
    hierarchy.cpp
    node.cpp
)
add_library(sub::hierarchy ALIAS ${PROJECT_NAME})
