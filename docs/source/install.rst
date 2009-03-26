Installation
============

Requirements
------------

You need to have the following software properly installed in order to
build *PETSc for Python*:

* Any MPI_ implementation [#]_ (e.g., MPICH_ or `Open MPI`_), 
  built with shared libraries.

* PETSc_ 2.3.2/2.3.3/3.0.0 release, built with shared libraries.

* Python_ 2.4/2.5/2.6 [#]_.

* NumPy_ package.

.. [#] Unless you have appropiatelly configured and built PETSc
       PETSc without MPI (configure option ``--with-mpi=0``).

.. [#] You may need to use a parallelized version of the Python
       interpreter with some MPI-1 implementations (e.g. MPICH1).

.. include:: links.txt

Using **setuptools**
--------------------

If you already have a working PETSc you can take advantage of
setuptools's :program:`easy_install` command::

   $ export PETSC_DIR=/path/to/petsc
   $ export PETSC_ARCH=linux-gnu # may not be requiered

   $ easy_install petsc4py

Using **distutils**
-------------------

Downloading
^^^^^^^^^^^

The *PETSc for Python* package is available for download at the
project website generously hosted by Google Code. You can use
:program:`wget` to get a release tarball::

   $ wget http://petsc4py.googlecode.com/files/petsc4py-X.X.X.tar.gz

Building
^^^^^^^^

After unpacking the release tarball::

   $ tar -zxf petsc4py-X.X.X.tar.gz
   $ cd petsc4py-X.X.X

the distribution is ready for building.

Some environmental configuration is needed to inform the location of
PETSc. You can set (using :command:`setenv`, :command:`export` or what
applies to you shell or system) the environmental variables
:envvar:`PETSC_DIR`, and :envvar:`PETSC_ARCH` indicating where you
have built/installed PETSc::

   $ export PETSC_DIR=/usr/local/petsc/3.0.0
   $ export PETSC_ARCH=linux-gnu

Alternatively, you can edit the file :file:`setup.cfg` and provide the
required information below the ``[config]`` section::

   [config]
   petsc_dir  = /usr/local/petsc/3.0.0
   petsc_arch = linux-gnu
   ...

Finally, you can build the distribution by typing::

   $ python setup.py build

Installing
^^^^^^^^^^

After building, the distribution is ready for installation. 

You can do a site-install type::

   $ python setup.py install

or, in case you need root privileges::

   $ su -c 'python setup.py install'

This will install the :mod:`petsc4py` package in the standard location
:file:`{prefix}/lib/python{X}.{X}/site-packages`.

You can also do a user-install type::

   $ python setup.py install --home=$HOME

This will install the :mod:`petsc4py` package in the standard location
:file:`$HOME/lib/python` (or perhaps :file:`$HOME/lib64/python`). This
location should be listed in the :envvar:`PYTHONPATH` environmental
variable.
