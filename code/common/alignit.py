"""align-it pharmacophore reader and writer abstractions
"""
from collections import namedtuple

PharmProp = namedtuple("PharmProp", "name x y z a n nx ny nz")
MolPharmacophores = namedtuple("MolPharmacophores", "name phar")

def read_pharm_property(line):
    name, x, y, z, a, has_normal, nx, ny, nz = line.strip().split("\t")[:9]
    return PharmProp(name, float(x), float(y), float(z), float(a), int(has_normal), float(nx), float(ny), float(nz))

def read_pharm_set(lines):
    name = None
    descriptors = []
    for line in lines:
        line = line.strip()
        if len(line) < 1:
            continue  # just in case there are empty lines
        if line.startswith("NAME"):
            name = line[4:].strip()
            descriptors = []
        elif line == "$$$$":
            yield MolPharmacophores(name, descriptors)
        else:
            descriptors.append(read_pharm_property(line))



