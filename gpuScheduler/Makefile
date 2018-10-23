.PHONY: all scheduler clear

all:
	reset
	cd .. && make all
run:
	cd .. && make clear
	cd .. && make all
	reset
	cd ../simulator && ./simulator.out
	./gpuscheduler.out -s 40
scheduler:
	cd .. && make scheduler
clear:
	cd .. && make clear
