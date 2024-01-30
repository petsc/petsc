import shutil
import re
import os

def update_htmlmap_links(builder,htmlmap_filename):
    """ Update manualpage links in an htmlmap file for use with Sphinx HTML builders html and dirhtml

        This converts file names in the htmlmap file (ending with .md) to the locations of the
        file that Sphinx generates.
    """

    if builder.name == "dirhtml":
        postfix = "/"
    elif builder.name == "html":
        postfix = ".html"
    else:
        raise Exception("Unsupported builder named %s" % builder.name)

    with open(htmlmap_filename+'_modified', "w") as htmlmap_file_modified, open(htmlmap_filename, "r") as htmlmap_file:
        pattern = re.compile(r".*\+\+\+\+man\+(.*)$")  # Match URL in group
        for line in htmlmap_file.readlines():
            match = re.match(pattern, line)
            if match:
                url = match.group(1)
                if url.startswith("manualpages"):
                    line = os.path.splitext(line)[0] + postfix + "\n"
            htmlmap_file_modified.write(line)
    os.rename(htmlmap_filename + '_modified',htmlmap_filename)
