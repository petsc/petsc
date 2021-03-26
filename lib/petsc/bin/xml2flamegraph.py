"""Convert a PETSc XML file into a Flame Graph input file."""

import argparse
import os
import sys

try:
    from lxml import objectify
except ImportError:
    sys.exit("Import error: lxml must be installed. Try 'pip install lxml'.")


class Event:

    def __init__(self, stack, time):
        self._stack = stack
        self._time = time  # must be in seconds

    def to_flamegraph_str(self):
        """Return the event as a string in the Flame Graph format.

        For example: 
            'main event;event1;event4 123'

        The event duration is given in microseconds because only integer
        values are accepted by speedscope.
        """
        return f"{';'.join(self._stack)} {int(self._time*1e6)}"


def parse_xml(input_file):
    """Traverse the XML and return a list of events."""
    root = objectify.parse(input_file).find("//timertree")
    total_time = root.find("totaltime")

    events = []
    for event in root.findall(".//event"):
        name = event.name.text
        # Skip 'self' and 'other' events.
        if name == "self" or name.endswith(": other-timed"): continue

        stack = [name]
        for ancestor in event.iterancestors():
            if ancestor.tag != "event":
                continue
            stack.append(ancestor.name.text)
        # The callstack needs to start from the root.
        stack.reverse()

        # The time can either be stored under 'value' or 'avgvalue'.
        if hasattr(event.time, "value"):
            time = event.time.value
        elif hasattr(event.time, "avgvalue"):
            time = event.time.avgvalue
        else:
            raise AssertionError

        # The times in the XML file are given as percentages, we need to 
        # convert them to actual values in seconds.
        time *= total_time / 100
        events.append(Event(stack, time))
    return events


def write_flamegraph_str(events, outfile):
    """Output the list of events to a file."""
    flamegraph_str = "\n".join(event.to_flamegraph_str() for event in events)
    with open(outfile, "w") as f:
        f.write(flamegraph_str)


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
    if not args.outfile.endswith(".txt"):
        raise ValueError("Output file must be a text file.")


def main():
    args = parse_args()
    check_args(args)

    events = parse_xml(args.infile)
    write_flamegraph_str(events, args.outfile)


if __name__ == "__main__":
    main()
