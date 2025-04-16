import os
import copy
import argparse
import json
import itertools
import sys
from functools import cmp_to_key


parameters = ["solver", "sigma", "nfac", "conv"]
exclude = "metadata"


markers = ("square", "o", "star", "triangle", "pentagon", "otimes", "oplus", "square*", "triangle*", "pentagon*")
colors = ("blue", "cyan", "gray", "green", "lime", "magenta", "olive", "orange", "pink", "purple", "red", "teal", "violet", "white", "yellow", "black", "brown")
shade = 80


def pretty_print(graphs):
    strs = []
    for g in graphs:
        f = ""
        for k in sorted(graphs[g], key=float):
            f += f"({k},{graphs[g][k]})"
        strs.append((g, f))
    return strs


def pretty_print_3d(graphs):
    strs = []

    def cmp(x, y):
        x = (float(x[0]), float(x[1]))
        y = (float(y[0]), float(y[1]))
        if x[1] < y[1]:
            return -1
        elif x[1] > y[1]:
            return 1
        else:
            if x[0] < y[0]:
                return -1
            elif x[0] > y[0]:
                return 1
            else:
                return 0

    for g in graphs:
        f = ""
        for k in sorted(graphs[g], key=cmp_to_key(cmp)):
            f += f"({k[0]}, {k[1]},{graphs[g][k]})"
        strs.append((g, f))
    return strs


def get_graphs_from_jsons(jsons, x_axis, y_axis, z_axis, parameters, exclude):
    final_graph = {}
    for json_file in jsons:
        if os.stat(json_file).st_size == 0:
            print(f"{json_file} is empty. Skipping..", file=sys.stderr)
            continue
        with open(json_file, 'r') as f:
            d = json.load(f)
        new_graphs, _ = get_graphs_from_dict(d, x_axis, y_axis, z_axis, parameters, exclude)
        for graph in new_graphs:
            if graph[0] in final_graph:
                if not set(final_graph[graph[0]].keys()).isdisjoint(set(graph[1].keys())):
                    print(f"WARNING:Duplicate setting found. {json_file=}; {graph[1]=}", file=sys.stderr)
                    print("First graph: ", file=sys.stderr)
                    print(final_graph[graph[0]], file=sys.stderr)
                    print("Second graph:", file=sys.stderr)
                    print(graph[1], file=sys.stderr)
                    for k in graph[1].keys():
                        if k in final_graph[graph[0]]:
                            if not final_graph[graph[0]][k] == graph[1][k]:
                                print(f"WARNING:CONFLICTING DATA for {k=}:", file=sys.stderr)
                                print(f"  Original:{final_graph[graph[0]][k]}; new: {graph[1][k]}", file=sys.stderr)
                            else:
                                print(f" Data agrees for {k=}; both {final_graph[graph[0]][k]}", file=sys.stderr)
                        else:
                            final_graph[graph[0]][k] = graph[1][k]
                else:
                    final_graph[graph[0]] |= graph[1]
            else:
                final_graph[graph[0]] = graph[1]
        print(f"Loaded {json_file}.", file=sys.stderr)
    # print(final_graph)
    # print()
    return final_graph


def get_graphs_from_dict(d, x_axis, y_axis, z_axis, categories, exclude):
    if len(categories) == 0:
        return [((), d[y_axis])], False
    sub_cats = copy.copy(categories)
    cur_cat = sub_cats.pop(0)
    is_xorz_axis = cur_cat == x_axis or cur_cat == z_axis
    found_xorz = False
    if is_xorz_axis:
        graphs_dict = {}
    else:
        new_graphs = []
    for cur_cat_v in d:
        if cur_cat_v in exclude:
            continue
        graphs, found_xorz_sub = get_graphs_from_dict(d[cur_cat_v], x_axis, y_axis, z_axis, sub_cats, exclude)
        found_xorz |= found_xorz_sub
        for g in graphs:
            if is_xorz_axis:
                if not found_xorz:
                    if g[0] in graphs_dict:
                        graphs_dict[g[0]][cur_cat_v] = g[1]
                    else:
                        graphs_dict[g[0]] = {cur_cat_v: g[1]}
                else:
                    left = cur_cat == x_axis
                    for key in g[1]:
                        if g[0] not in graphs_dict:
                            graphs_dict[g[0]] = {}
                        if left:
                            graphs_dict[g[0]][(key, cur_cat_v)] = g[1][key]
                        else:
                            graphs_dict[g[0]][(cur_cat_v, key)] = g[1][key]
            else:
                new_graph = (tuple([cur_cat_v] + list(g[0])), g[1])
                new_graphs.append(new_graph)
    if is_xorz_axis:
        return list(graphs_dict.items()), True
    else:
        return new_graphs, found_xorz


def find_jsons(path, use_all, nosubdirs):
    jsons = []
    jn = os.path.join
    if nosubdirs:
        cur = path
        jsons_loc = [jn(cur, name) for name in os.listdir(cur) if os.path.isfile(
            jn(cur, name)) and name.endswith(".json") and not name.endswith(".part.json")]
        jsons += jsons_loc
    else:
        subdirs = [name for name in os.listdir(path)
                   if os.path.isdir(jn(path, name))]
        for subd in subdirs:
            if len(subdirs) > 1 and not use_all:
                yn = input(f"Use {subd}? ").strip().lower()
                if yn != "y" and yn != "yes":
                    print(f"Skipping {subd}.", file=sys.stderr)
                    continue
            print(f"Using {subd}.", file=sys.stderr)
            cur = jn(path, subd)
            jsons_loc = [jn(cur, name) for name in os.listdir(cur) if os.path.isfile(
                jn(cur, name)) and name.endswith(".json") and not name.endswith(".part.json")]
            jsons += jsons_loc
    return jsons


def print_pgf(pretty, x_axis, z_axis, parameters, skip_colors):
    f_up = ""
    f_lower = "\\legend{"
    compact = ""

    def to_desc(g):
        res = ""
        if z_axis is None:
            assert len(g) == len(parameters) - 1
        else:
            assert len(g) == len(parameters) - 2
        it_c = iter(parameters)
        it_g = iter(g)
        for _ in range(len(parameters)):
            c = next(it_c)
            if c == x_axis or c == z_axis:
                continue
            gi = next(it_g)
            res += f"{c}={gi} "
        res = res[:-1]
        return res

    cols = itertools.cycle(colors)
    mars = itertools.cycle(markers)
    for _ in range(skip_colors):
        next(cols)
        next(mars)
    for (g, f) in pretty:
        f_up += "\\addplot[\ncolor=" + next(cols) + "!" + str(shade) + "!black,\nmark=" + next(mars) + ",\n]\ncoordinates {\n" + f + "\n};\n\n"
        f_lower += f"{to_desc(g)}, "
        compact += f"%{to_desc(g)}\n{f}\n\n"
    f_lower = f_lower[:-2] + "}"
    print(f_up)
    print(f_lower)
    return compact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=str, required=True)
    parser.add_argument("-z", type=str, required=False, default=None)
    parser.add_argument("-y", type=str, required=True, nargs="+")
    parser.add_argument("--parameters", "-p", type=str, required=False, default=parameters, nargs="+")
    parser.add_argument("--exclude", "-e", type=str, required=False, default=exclude, nargs="+")
    parser.add_argument("--use-all-jsons", action="store_true")
    parser.add_argument("--no-subdirs", action="store_true")
    parser.add_argument("--sorted", action="store_true")
    parser.add_argument("--skip-colors", default=0, type=int, required=False)
    parser.add_argument("path", type=str, nargs='+')
    args = parser.parse_args()
    jsons = []
    for p in args.path:
        if not os.path.isdir(p):
            assert p.endswith(".json"), "Either pass a json or a directory with .json's."
            jsons += [p,]
        else:
            jsons += find_jsons(p, args.use_all_jsons, args.no_subdirs)
    if len(jsons) == 0:
        print("No JSONs", file=sys.stderr)
        exit(0)
    print("="*10 + "Dict" + "="*10 + "\n", file=sys.stderr)
    compact_print = ""
    for y_param in args.y:
        graphs = get_graphs_from_jsons(jsons, args.x, y_param, args.z, args.parameters, args.exclude)
        print()
        if args.sorted:
            graphs = dict(sorted(graphs.items(), key=lambda x: float(x[0][1])))
        if args.z is not None:
            fs = pretty_print_3d(graphs)
        else:
            fs = pretty_print(graphs)
        print("="*10 + "PGF" + "="*10 + "\n", file=sys.stderr)
        compact_print += "="*10 + y_param + "="*10 + "\n"
        compact_print += print_pgf(fs, args.x, args.z, args.parameters, args.skip_colors)
    print("="*10 + "COMPACT" + "="*10 + "\n", file=sys.stderr)
    print(compact_print, file=sys.stderr)


if __name__ == "__main__":
    main()
