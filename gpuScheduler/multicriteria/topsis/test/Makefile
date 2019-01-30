all:
	nvcc topsis_test.cu ../topsis.cu ../../../json.cpp -O2 -std=c++14 -Xptxas -O2 -use_fast_math -lineinfo -o topsis
debug:
	nvcc topsis_test.cu ../topsis.cu ../../../json.cpp -g -std=c++14 -Xptxas -O2 -use_fast_math -lineinfo -o topsis
clean:
	rm -f topsis
