# Network-Awere GPU Accelerated Containers Scheduler (GPUACS/VIMAM)

## Introduction

GPUACS is a Linux software to simulate a cloud scheduler. The main features of the GPUACS are the GPU acceleration provided the ranking and virtual network ranking algorithms in online and offline scheduling policies.

This software is well in two articles:

In [Master thesis (PT-br only)](master_thesis.pdf), an English version is under production.

In `GLOBCOM2019` article: https://ieeexplore.ieee.org/document/9013128/

## Dependencies

The GPUACS requires CUDA and C++17 installed in your machine.
All the others dependencias are inside the thirdparty folder.
- rapidjson: https://rapidjson.org/
- spdlog: https://github.com/gabime/spdlog
- catch2: https://github.com/catchorg/Catch2
- clara: https://github.com/catchorg/Clara

## Compilation

The compilation is made through [CMAKE](https://github.com/Kitware/CMake).

```bash
mkdir build && cd build
cmake ../scheduler
cmake --build .
```

## How to run the scheduler

```bash
#Go to the bin folder
cd bin
#To run the scheduler just run:
./gpu_scheduler.out args
#To check the available options:
./gpu_scheduler.out -h
```

## Modularity

The GPUACS is a modular scheduler that allows you to implement your algorithms and plug then into the scheduler. To do it, you need to import our respective interface and dependencies into your code and include your code into the gpu_scheduler.cu file.

For example, if you have a round-robin ranking algorithm and want to plug it into GPUACS ranking algorithms, to accomplish that you will need:

1- move your file into the allocator/rank_algorithms/user_defined/ folder.
2- import the interface and respective dependencies.
```cpp
#include "../rank.hpp" //ranking algorithms interface
#include "../../free.hpp" //scheduler utils used to free resources
#include "../../utils.hpp" // scheduler utils used to check resources bounds
``` 
3- import your file into the gpu_scheduler.cu
```cpp
#include "allocator/rank_algorithms/user_defined/$your_file"
```
4- change the allowed arguments inside gpu_scheduler.cu file.
```cpp
//include your ranking algorithm
clara::detail::Opt( rank_method, "rank method") ["-r"]["--rank"] ("What is the rank method? [ ahp | (default) ahpg | topsis | best-fit (bf) | worst-fit (wf) | user_defined (INCLUDE THIS)")

if( rank_method == "ahpg") {
    rank = new AHPG();
}else if(rank_method == "ahp" ) {
    rank = new AHP();
}else if(rank_method=="topsis") {
    rank = new TOPSIS();
} else if(rank_method=="bf" || rank_method=="best-fit") {
    rank = new BestFit();
} else if(rank_method=="wf" || rank_method=="worst-fit") {
    rank = new WorstFit();
} else if(rank_method=="user_defined_algorithm") {
    rank = new USER_DEFINED_ALGORITHM //INCLUDE THIS
}
} else{
    SPDLOG_ERROR("Invalid rank method");
    exit(0);
}
```

And that's it, now your ranking algorithm is ready to work inside the GPUACS scheduler.

You can add algorithms inside the other scheduler's modules in the same way.

## Configuration files

All the scheduler settings can be changed through editing the JSON files inside the scheduler folder.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update the tests as appropriate.

## License
This software is under Mozilla Public License 2.0 license
