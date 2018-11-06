all:
	reset
	g++ hierarchy_tests.cpp ../ahp.cpp ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -fconcepts -std=c++17 -o ahp
	#g++ hierarchy_tests.cpp ../ahp.cpp ../../json.cpp -fconcepts -std=c++17 -o ahp
	#nvcc hierarchy_gpu.cpp ../ahpg.cu ../../json.cpp -o ahpg
	reset
	./ahp
gpu:
	nvcc hierarchy_gpu_test.cpp ../ahpg.cu ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -o ahpg
ahp:
	g++ hierarchy_tests.cpp ../ahp.cpp ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -fconcepts -std=c++17 -o ahp
pure1:
	g++ pure_test1.cpp ../ahp.cpp ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -fconcepts -std=c++17 -o ahp -g
	reset
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./ahp
pure2:
	g++ pure_test2.cpp ../ahp.cpp ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -fconcepts -std=c++17 -o ahp -g
		reset
		valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./ahp
clean:
	rm -f ahpg ahp
