## Sphinx Documentation

We use Sphinx to generate some documentation. This makes it easy to generate
documentation in various formats, and can be integrated with services like
ReadTheDocs to give web-based documentation which should be easy to find,
maintain, and contribute to.

Some of the files here were created with the help of `sphinx-quickstart` and
then edited.

To build and view locally, make sure that you have a suitable version of python,
and the required modules. You can determine that from what we tell ReadTheDocs
about (at the time of this writing, see `.readthedocs.yml` in the root directory,
and `requirements.txt` here). Then, something similar to

    make html
    firefox _build/html/index.html
