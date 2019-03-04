#!/usr/bin/env python

import argparse
import io
import random
import numpy
import math
from json import dump, dumps, loads, JSONEncoder, JSONDecoder
import pickle
from dataModel import PythonObjectEncoder, ContainerLink, Container, Task, Tasks

#CPU=24
#RAM=256
#BW_CONTAINER=10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--requests',type=int,action='store',dest='requests',
                        default=100,
                        help='number requests')
    parser.add_argument('-i','--intervals',type=int,action='store',dest='intervals',
                        default='1000',
                        help='number of intervals (ms)')
    parser.add_argument('-d','--distribution',type=str,action='store',dest='distribution',
                        default='uniform',
                        help='uniform poisson')
    parser.add_argument('-s', '--size', type=int, action='store',dest='size',default='10',
                        help='size of each request')
    parser.add_argument('-p','--pod',type=float,action='store',dest='pod',
                        default='0.5',
                        help='%% of conteiners in pod')
    parser.add_argument('-b','--bandwidth',type=float,action='store',dest='bandwidth',
                        default='0.25',
                        help='max bandwidth of between containers')
    parser.add_argument('-l','--duration',type=int,action='store',dest='duration',
                        default='100',
                        help='max duration of a task')
    parser.add_argument('-o','--output',type=str,action='store',dest='output',
                        default='requests.json',
                        help='output file')
    parser.add_argument('-c','--cpu',type=float,action='store',dest='CPU',
                        default=2,
                        help='max CPU size container')
    parser.add_argument('-m','--memory',type=float,action='store',dest='RAM',
                        default=4,
                        help='max RAM size container')

    return parser.parse_args()

def get_uniform_int(min_value, max_value):
    return random.randint(min_value, max_value + 1)

def get_uniform_float(min_value, max_value):
    return random.uniform(min_value, max_value + 1)

if __name__ == '__main__':
    args = parse_args()

    task_id = 1
    pod_id = 1
    tasks = Tasks()
    for r in range(1, args.requests + 1):
        if args.distribution == 'uniform':
            submission = get_uniform_int(1, args.intervals)
        elif args.distribution == 'poisson':
            print 'Please someone implement me'
            break
        else:
            print 'Invalid distribution!'
            break
        duration = get_uniform_int(1, args.duration)
        t = Task(task_id, submission, duration)

        cont_id = 1
        cpods = math.ceil(args.size * args.pod)
        for c in range(1, args.size + 1):
            min_cpu = get_uniform_float(1, args.CPU)
            max_cpu = get_uniform_float(min_cpu, args.CPU)
            min_memory = get_uniform_float(1, args.RAM)
            max_memory = get_uniform_float(min_memory, args.RAM)
            pod = abs(cpods)
            cpods = cpods - 1
                
            if cpods == 0:
                pod_id = pod_id + 1
                cpods = math.ceil(args.size * args.pod)


            c = Container(cont_id, min_cpu, max_cpu, min_memory, max_memory, 0, 0, pod)
            t.addContainer(c)

            cont_id = cont_id + 1

        for i in range(1, args.size + 1):
            for j in range(i + 1, args.size + 1):
                bw_min = get_uniform_float(1, args.bandwidth)
                bw_max = get_uniform_float(bw_min, args.bandwidth)
                cl = ContainerLink(str(i) + '-' + str(j), i, j, bw_min, bw_max)
                t.addLink(cl)

        tasks.addTask(t)
        task_id = task_id + 1

    with open(args.output, 'w') as f:
        f.write(dumps(tasks.__dict__, indent=4))
