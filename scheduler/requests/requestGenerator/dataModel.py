#!/usr/bin/env python

import argparse
import io
import random
import numpy
import math
from json import dump, dumps, loads, JSONEncoder, JSONDecoder
import pickle

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, unicode, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}

class ContainerLink:
    def __init__(self, name, source, destination, bandwidth_min, bandwidth_max):
        self.name = name
        self.source = source
        self.destination = destination
        self.bandwidth_min = float(bandwidth_min)
        self.bandwidth_max = float(bandwidth_max)

class Container:
    def __init__(self, name, vcpu_min, vcpu_max, ram_min, ram_max, epc_min, epc_max, pod):
        self.name = name
        self.vcpu_min = float(vcpu_min)
        self.vcpu_max = float(vcpu_max)
        self.ram_min = float(ram_min)
        self.ram_max = float(ram_max)
        self.epc_min = float(epc_min)
        self.epc_max = float(epc_max)
        self.pod = int(pod)

class Task:
    def __init__(self, name, submission, duration):
        self.id = name
        self.submission = float(submission)
        self.duration = float(duration)
        self.containers = []
        self.links = []

    def addContainer(self, container):
        self.containers.append(container.__dict__)

    def addLink(self, link):
        self.links.append(link.__dict__)

class Tasks:
    def __init__(self):
        self.tasks = []

    def addTask(self, task):
        self.tasks.append(task.__dict__)
