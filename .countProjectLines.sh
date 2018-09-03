find . -name '*.cpp' -o -name '*.hpp' -o -name "*.cu" -o -name "*.cuh" -o -name "thirdparty" -prune  | xargs wc -l
