import shutil
import re
import os

from build_classic_docs import HTMLMAP_DEFAULT_LOCATION


def update_htmlmap_links(builder,
                         htmlmap_filename=HTMLMAP_DEFAULT_LOCATION,
                         htmlmap_modified_filename=HTMLMAP_DEFAULT_LOCATION + "_modified"):
    """ Update manualpage links in an htmlmap file for use with Sphinx HTML builders

        This converts source files (.md) to the locations of the resulting .html pages
        or directories (which a web server will redirect to the index.html inside).
    """

    if builder.name == "dirhtml":
        postfix = "/"
    elif builder.name == "html":
        postfix = ".html"
    else:
        raise Exception("Unsupported builder named %s" % builder.name)

    with open(htmlmap_modified_filename, "w") as htmlmap_modified_file, open(htmlmap_filename, "r") as htmlmap_file:
        pattern = re.compile(".*\+\+\+\+man\+(.*)$")  # Match URL in group
        for line in htmlmap_file.readlines():
            match = re.match(pattern, line)
            if match:
                url = match.group(1)
                if url.startswith("manualpages"):
                    line = os.path.splitext(line)[0] + postfix + "\n"
            htmlmap_modified_file.write(line)
