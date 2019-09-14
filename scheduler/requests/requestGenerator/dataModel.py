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
        self.bandwidth_min = round(float(bandwidth_min),5)
        self.bandwidth_max = round(float(bandwidth_max),5)

class Container:
    def __init__(self, name, vcpu_min, vcpu_max, ram_min, ram_max, epc_min, epc_max, pod):
        self.name = name
        self.vcpu_min = round(float(vcpu_min),5)
        self.vcpu_max = round(float(vcpu_max),5)
        self.ram_min = round(float(ram_min),5)
        self.ram_max = round(float(ram_max),5)
        self.epc_min = round(float(epc_min),5)
        self.epc_max = round(float(epc_max),5)
        self.pod = int(pod)

class Task:
    def __init__(self, name, submission, duration, deadline):
        self.id = name
        self.submission = float(submission)
        self.duration = float(duration)
        self.deadline = float(deadline)
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
