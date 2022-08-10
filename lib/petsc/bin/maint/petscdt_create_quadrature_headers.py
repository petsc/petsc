"""
Create header files with:

- triangle quadrature rules from the supplementary data in https://doi.org/10.1016/j.camwa.2015.03.017
- tetrahedral quadrature rules from the supplementary data in https://doi.org/10.1002/nme.6528
"""

import os
import re
import glob
import gmpy2
from gmpy2 import mpfr
import pprint

gmpy2.get_context().precision = 113 # Generate quad precision headers

def triangle_rules(rules_path):
    """
    Gather and format quadrature rule data from the unpacked zip file found
    at https://doi.org/10.1016/j.camwa.2015.03.017
    """

    compact_folder = os.path.join(rules_path, "compact", "tri")
    expanded_folder = os.path.join(rules_path, "expanded", "tri")

    def biunit_to_bary(x):
        b = [mpfr(0), mpfr(0), mpfr(0)]
        btob = [
                [ 0.5, 0.0, 0.5],
                [ 0.0, 0.5, 0.5],
                [-0.5,-0.5, 0.0]
                ]
        for i in range(3):
            for j in range(3):
                b[i] += btob[i][j] * x[j]
        return b

    rules = {}
    for file in os.listdir(compact_folder):
        name = os.path.basename(file)
        descrip = os.path.splitext(name)[0]
        parts = descrip.split('-')
        power = int(parts[0])
        num_points = int(parts[1])
        counts = ()
        with open(os.path.join(compact_folder,file),"r") as f:
            counts = tuple(int(s) for s in re.split('\s+',f.readline().strip()))
        rule = {}
        rule['num_nodes'] = num_points
        rule['counts'] = counts
        rule['nodes'] = [[], [], []]
        rule['weights'] = []
        with open(os.path.join(expanded_folder,file),"r") as f:
            for (i,(count, multiplicity)) in enumerate(zip(counts, (1,3,6))):
                for c in range(count):
                    for m in range(multiplicity):
                        x, y, weight = tuple(mpfr(s) for s in re.split('\s+',f.readline().strip()))
                        if m == 0:
                            rule['weights'].append(weight)
                            if i == 0:
                                rule['nodes'][i].append([mpfr(1) / mpfr(3)])
                            elif i == 1:
                                X = [x, y, mpfr(1)]
                                B = biunit_to_bary(X)
                                nodes = sorted(B)
                                diffs = [b - a for (a,b) in zip(nodes[:-1],nodes[1:])]
                                if diffs[0] > diffs[1]: # the first is the odd one out
                                    rule['nodes'][i].append([nodes[-1], nodes[0]])
                                else: # the last is the odd one out
                                    rule['nodes'][i].append([nodes[0], nodes[-1]])
                            else:
                                X = [x, y, mpfr(1)]
                                B = biunit_to_bary(X)
                                rule['nodes'][i].append(B)
        rules[power] = rule
    rules["type"] = "triangle"
    rules["prefix"] = "WVTri"
    rules["url"] = "https://doi.org/10.1016/j.camwa.2015.03.017"
    return rules


def tetrahedron_rules(rules_path):
    """
    Gather and format quadrature rule data from the unpacked zip file found
    at https://doi.org/10.1002/nme.6528
    """

    rules = {}
    rules_path = os.path.join(rules_path, "cubatures_tet_sym_repository")
    for file in glob.glob(os.path.join(rules_path, "*_compact.txt")):
        name = os.path.basename(file)
        descrip = os.path.splitext(name)[0]
        parts = re.search('cubature_tet_sym_p([0-9]+)_n([0-9]+)_compact', descrip)
        power = int(parts.group(1))
        num_points = int(parts.group(2))
        counts = ()
        with open(os.path.join(rules_path,file),"r") as f:
            count_regex = re.findall('=([0-9]+)', f.readline())
            counts = tuple(int(s) for s in count_regex)
        rule = {}
        rule['num_nodes'] = num_points
        rule['counts'] = counts
        rule['nodes'] = [[], [], [], [], []]
        rule['weights'] = []
        exname = name.replace('compact','expand_baryc')
        with open(os.path.join(rules_path, exname),"r") as f:
            for (i,(count, multiplicity)) in enumerate(zip(counts, (1,4,6,12,24))):
                for c in range(count):
                    for m in range(multiplicity):
                        w, x, y, z, weight = tuple(mpfr(s) for s in re.split('\s+',f.readline().strip()))
                        if m == 0:
                            rule['weights'].append((weight * 4) / 3)
                            if i == 0: # (a, a, a, a)
                                rule['nodes'][i].append([w])
                            elif i == 1: # (a, a, a, b)
                                nodes = sorted([w, x, y, z])
                                diffs = [b - a for (a,b) in zip(nodes[:-1],nodes[1:])]
                                if diffs[0] > diffs[-1]: # first index is the odd one out
                                    rule['nodes'][i].append([nodes[-1], nodes[0]])
                                else: # last index is the odd one out
                                    rule['nodes'][i].append([nodes[0], nodes[-1]])
                            elif i == 2: # (a, a, b, b)
                                nodes = sorted([w, x, y, z])
                                rule['nodes'][i].append([nodes[0], nodes[-1]])
                            elif i == 3: # (a, a, b, c)
                                nodes = sorted([w, x, y, z])
                                diffs = [b - a for (a,b) in zip(nodes[:-1],nodes[1:])]
                                j = 0
                                for k in range(len(diffs)):
                                    if diffs[k] < diffs[j]:
                                        j = k
                                if j == 0: # 0 and 1 are the same
                                    rule['nodes'][i].append([nodes[0], nodes[2], nodes[3]])
                                elif j == 1: # 1 and 2 are the same
                                    rule['nodes'][i].append([nodes[1], nodes[0], nodes[3]])
                                else: # 2 and 3 are the same
                                    rule['nodes'][i].append([nodes[2], nodes[0], nodes[1]])
                            else:
                                rule['nodes'][i].append([w, x, y, z])
        rules[power] = rule
    rules["type"] = "tetrahedron"
    rules["prefix"] = "JSTet"
    rules["url"] = "https://doi.org/10.1002/nme.6528"
    return rules


def create_header(rules):
    powers = [i for i in filter(lambda x: type(x) == int, rules.keys())]
    max_power = max(powers)
    filename = f"petscdt{rules['type'][:3]}quadrules.h"
    guard = filename.replace('.','_').upper()
    lines = []
    lines.append(f"""// Minimal symmetric quadrature rules for a {rules['type']} from {rules['url']}
// This file was generated by lib/petsc/bin/maint/petscdt_create_quadrature_headers.py
#if !defined({guard})
#define {guard}

#include <petscsys.h>
""")
    printer_wide = pprint.PrettyPrinter(width=260)
    printer_narrow = pprint.PrettyPrinter(width=58)
    prefix = f"PetscDT{rules['prefix']}Quad"
    for key in sorted(powers):
        rule = rules[key]
        lines.append(f"static const PetscReal {prefix}_{key}_weights[] = {{");
        weights = printer_narrow.pformat(["PetscRealConstant({0:e})".format(w) for w in rule['weights']]).replace('[',' ').replace("'","").strip(']')
        lines.append(f"{weights}")
        lines.append("};")
        lines.append("")
        lines.append(f"static const PetscReal {prefix}_{key}_orbits[] = {{");
        for (i,nodes) in enumerate(rule['nodes']):
            for (j,node) in enumerate(nodes):
                fmt = printer_wide.pformat(["PetscRealConstant({0:e})".format(w) for w in node]).replace('[',' ').replace("'","")
                if sum([len(s) for s in rule['nodes'][i+1:]]) == 0 and j == len(nodes) - 1:
                    fmt = re.sub('\s*]','',fmt)
                else:
                    fmt = re.sub('\s*]',',',fmt)
                lines.append(fmt)
        lines.append("};")
        lines.append("")

    lines.append(f"static const PetscInt {prefix}_max_degree = {max_power};")
    lines.append("")
    degree_to_power = [-1 for i in range(max_power+1)]
    for i in range(max_power, -1, -1):
        if i in rules.keys():
            degree_to_power[i] = i
        else:
            degree_to_power[i] = degree_to_power[i+1]

    num_nodes = [rules[degree_to_power[d]]['num_nodes'] for d in range(max_power+1)]
    lines.append(f"static const PetscInt {prefix}_num_nodes[] = {{")
    lines.append(f"{printer_narrow.pformat(num_nodes).replace('[',' ').strip(']')}")
    lines.append("};")
    lines.append("")

    num_orbits = [rules[degree_to_power[d]]['counts'] for d in range(max_power+1)]
    lines.append(f"static const PetscInt {prefix}_num_orbits[] = {{")
    lines.append(f"{printer_narrow.pformat(num_orbits).replace('[',' ').strip(']').replace('(','').replace(')','')}")
    lines.append("};")
    lines.append("")

    for suff in ["weights", "orbits"]:
        lines.append(f"static const PetscReal *{prefix}_{suff}[] = {{");
        for i in range(max_power+1):
            if i == max_power:
                lines.append(f"  {prefix}_{degree_to_power[i]}_{suff}");
            else:
                lines.append(f"  {prefix}_{degree_to_power[i]}_{suff},");
        lines.append("};")
        lines.append("")

    lines.append(f"#endif // #define {guard}")

    with open(filename, "w") as f:
        f.writelines('\n'.join(lines) + '\n')


if __name__ == '__main__':
    import tempfile
    import requests
    from zipfile import ZipFile
    from tarfile import TarFile
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Create quadrature headers from published compact representations")
    parser.add_argument('--tri', type=pathlib.Path)
    parser.add_argument('--tet', type=pathlib.Path)
    args = parser.parse_args()

    def unzip_and_create_header(file, extracter, rules):
        with tempfile.TemporaryDirectory() as tmpdirname:
            _dir = os.path.join(tmpdirname, "rules")
            with extracter(file) as _archive:
                _archive.extractall(path=_dir)
            create_header(rules(_dir))

    if args.tri:
        unzip_and_create_header(args.tri, ZipFile, triangle_rules)

    if args.tet:
        unzip_and_create_header(args.tet, TarFile, tetrahedron_rules)

