""" Add redirect pages to preserve previous man page URLs """

import os

TEMPLATE = ('<!DOCTYPE html>\n'
            '<html>'
            '<head><meta http-equiv="refresh" content="0; url=%s" /></head>'
            '<body><a href="%s">Click here</a></body>'
            '</html>')


def _generate_page(name, location):
    redirect_filename = os.path.join(location, name + ".html")
    with open(redirect_filename, "w") as redirect_file:
        redirect_file.write(TEMPLATE % (name, name))


def add_man_page_redirects(root):
    """ Add redirecting .html files for all man pages.

    Assumes a particular directory structure relative to root.  For each
    subdirectory (manual page section) of root/docs/manualpages, enumerate all
    contained directories as names, and add links name.html which redirect to
    name/ (from which a web server will load name/index.html).
    """
    manualpages_dir = os.path.join(root, "docs", "manualpages")
    for mansec in os.listdir(manualpages_dir):
        subdirectory = os.path.join(manualpages_dir, mansec)
        if os.path.isdir(subdirectory):
            for name in os.listdir(subdirectory):
                if os.path.isdir(os.path.join(subdirectory, name)):
                    _generate_page(name, subdirectory)
