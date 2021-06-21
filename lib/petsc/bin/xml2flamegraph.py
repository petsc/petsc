"""Convert a PETSc XML file into a Flame Graph input file."""

import argparse
import os
import sys

try:
    from lxml import objectify
except ImportError:
    sys.exit("Import error: lxml must be installed. Try 'pip install lxml'.")


def parse_time(event):
    # The time can either be stored under 'value' or 'avgvalue'
    if hasattr(event.time, "value"):
        return event.time.value
    elif hasattr(event.time, "avgvalue"):
        return event.time.avgvalue
    else:
        raise AssertionError


def make_line(callstack, time, total_time):
    """The output time needs to be an integer for the file to be
    accepted by speedscope (speedscope.app). Therefore we output it in
    microseconds. It is originally a percentage of the total time
    (given in seconds).
    """
    event_str = ";".join(str(event.name) for event in callstack)
    time_us = int(time / 100 * total_time * 1e6)
    return f"{event_str} {time_us}"


def traverse_children(parent, total_time, callstack=None):
    if callstack == None:
        callstack = []

    # Sort the events into 'self' and child events
    self_events, child_events = [], []
    for event in parent.event:
        if event.name == "self" or str(event.name).endswith("other-timed"):
            self_events.append(event)
        else:
            child_events.append(event)

    lines = []
    if self_events:
        time = sum(parse_time(event) for event in self_events)
        lines.append(make_line(callstack, time, total_time))

    for event in child_events:
        # Check to see if event has any children. The latter check is for the
        # case when the <events> tag is present but empty.
        if hasattr(event, "events") and hasattr(event.events, "event"):
            callstack.append(event)
            lines.extend(traverse_children(event.events, total_time, callstack))
            callstack.pop()
        else:
            time = parse_time(event)
            lines.append(make_line(callstack+[event], time, total_time))
    return lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input XML file")
    parser.add_argument("outfile", type=str, help="Output file")
    return parser.parse_args()


def check_args(args):
    if not args.infile.endswith(".xml"):
        raise ValueError("Input file must be an XML file.")
    if not os.path.exists(args.infile):
        raise ValueError("The input file does not exist.")


def main():
    args = parse_args()
    check_args(args)

    root = objectify.parse(args.infile).find("//timertree")
    total_time = root.find("totaltime")
    lines = traverse_children(root, total_time)

    with open(args.outfile, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
