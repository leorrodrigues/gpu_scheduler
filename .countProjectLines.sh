find . -name '*.cpp' -o -name '*.hpp' -o -name "*.cu" -o -name "*.cuh" -o -name "thirdparty" -prune -o -name "build" -prune | xargs wc -l
