Installation
============

Requirements
------------

You need to have the following software properly installed in order to
build *PETSc for Python*:

* Any MPI_ implementation [#]_ (e.g., MPICH_ or `Open MPI`_), 
  built with **shared libraries**.

* PETSc_ 3.0.0 or 3.1 release, built with **shared libraries** [#]_.

* Python_ 2.4 to 2.7 and 3.1 to 3.2 [#]_.

* NumPy_ package.

.. [#] Unless you have appropiatelly configured and built PETSc
       PETSc without MPI (configure option :option:`--with-mpi=0`).

.. [#] In order to build PETSc with shared libraries, you have to pass
       :option:`--with-shared-libraries` option to PETSc's
       :program:`configure` script.

.. [#] You may need to use a parallelized version of the Python
       interpreter with some MPI-1 implementations (e.g. MPICH1).

.. include:: links.txt


Using **pip** or **easy_install**
---------------------------------

If you already have a working PETSc, set environment variables
:envvar:`PETSC_DIR` and perhaps :envvar:`PETSC_ARCH` to appropriate
values::

    $ export PETSC_DIR=/path/to/petsc
    $ export PETSC_ARCH=linux-gnu

.. note:: If you do not set these environment variables, the install
   process will attempt to download and install PETSc for you.

Now you can use :program:`pip`::

    $ [sudo] pip install [--user] petsc4py

Alternatively, you can use *setuptools* :program:`easy_install`
(deprecated)::

    $ [sudo] easy_install petsc4py


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

.. note:: **Mac OS X** users employing a Python distribution built
   with **universal binaries** may need to set the environment
   variables :envvar:`MACOSX_DEPLOYMENT_TARGET`, :envvar:`SDKROOT`,
   and :envvar:`ARCHFLAGS` to appropriate values. As an example,
   assume your Mac is running **Snow Leopard** on a **64-bit Intel**
   processor and you want to override the hard-wired cross-development
   SDK in Python configuration, your environment should be modified
   like this::

      $ export MACOSX_DEPLOYMENT_TARGET=10.6
      $ export SDKROOT=/
      $ export ARCHFLAGS='-arch x86_64'

Some environmental configuration is needed to inform the location of
PETSc. You can set (using :command:`setenv`, :command:`export` or what
applies to you shell or system) the environmental variables
:envvar:`PETSC_DIR`, and :envvar:`PETSC_ARCH` indicating where you
have built/installed PETSc::

   $ export PETSC_DIR=/usr/local/petsc
   $ export PETSC_ARCH=linux-gnu

Alternatively, you can edit the file :file:`setup.cfg` and provide the
required information below the ``[config]`` section::

   [config]
   petsc_dir  = /usr/local/petsc
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

You can also do a user-install type. Threre are two options depending
on the target Python version.

* For Python 2.6 and up::

      $ python setup.py install --user

* For Python 2.5 and below (assuming your home directory is available
  through the :envvar:`HOME` environment variable)::

      $ python setup.py install --home=$HOME

  and then add :file:`$HOME/lib/python` or :file:`$HOME/lib64/python`
  to your :envvar:`PYTHONPATH` environment variable.
