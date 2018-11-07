all:
ahp_hierarchy:
	g++ ahp_hierarchy.cpp ../ahp.cpp ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -fconcepts -std=c++17 -g -o ahp_hierarchy
	reset
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./ahp_hierarchy
ahp_major:
	g++ ahp_major.cpp ../ahp.cpp ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -fconcepts -std=c++17 -g -o ahp_major
	reset
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./ahp_major
ahp_car:
	g++ ahp_car.cpp ../ahp.cpp ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -fconcepts -std=c++17 -g -o ahp_car
	reset
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./ahp_car
ahp_files:
	g++ ahp_files.cpp ../ahp.cpp ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -fconcepts -std=c++17 -g -o ahp_files
	reset
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./ahp_files
ahpg_hierarchy:
	nvcc ahpg_hierarchy.cu ../ahpg.cu  ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -g -o ahpg_hierarchy
	reset
	cuda-memcheck ./ahpg_hierarchy
ahpg_files:
	nvcc ahpg_files.cu ../ahpg.cu  ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -g -o ahpg_files
	reset
	cuda-memcheck ./ahpg
clean:
	rm -f ahp_hierarchy ahp_major ahp_car ahp_files ahpg_hierarchy ahpg_files
