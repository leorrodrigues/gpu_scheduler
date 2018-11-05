all:
	g++ hierarchy.cpp ../ahp.cpp ../../json.cpp -fconcepts -std=c++17 -o ahp
	nvcc hierarchy_gpu.cpp ../ahpg.cu ../../json.cpp -o ahpg
gpu:
	nvcc hierarchy_gpu.cpp ../ahpg.cu ../../json.cpp -o ahpg
ahp:
	rm -f ahp
	reset
	g++ hierarchy.cpp ../ahp.cpp ../hierarchy/hierarchy_resource.cpp ../hierarchy/hierarchy.cpp ../hierarchy/node.cpp ../hierarchy/edge.cpp ../../json.cpp -fconcepts -std=c++17 -o ahp
clean:
	rm -f ahpg ahp
