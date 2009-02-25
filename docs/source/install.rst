Installation
============


Requirements
------------

* Any MPI distribution [#]_ (e.g., MPICH_ or `Open MPI`_), 
  built with shared libraries.

* PETSc_ 2.3.2/2.3.3/3.0.0 distribution, built with shared libraries.

* Python_ 2.4/2.5/2.6 [#]_.

* NumPy_ package (latest version).

.. [#] Unless you have appropiatelly configured and built PETSc
       without MPI.

.. [#] It is recommendable to use a parallelized version of the
       Python interpreter, but required if you want to use MPICH1.

.. include:: links.txt


Building
--------

Download and unpack the source distribution::

   $ wget http://petsc4py.googlecode.com/files/petsc4py-X.X.X.tar.gz
   $ tar -zxf petsc4py-X.X.X.tar.gz
   $ cd petsc4py-X.X.X

Some environmental configuration is needed to inform the location of
PETSc destribution. You can set (using ``setenv``, ``export`` or what
applies to you shell or system) the environmental variables
``PETSC_DIR`` and ``PETSC_ARCH`` indicating where you have
built/installed PETSc::

   $ export PETSC_DIR=/usr/local/petsc/3.0.0
   $ export PETSC_ARCH=linux-gnu

Alternatively, you can edit the file ``setup.cfg`` and provide the
required information below ``[config]`` section::

   [config]
   petsc_dir  = /usr/local/petsc/3.0.0
   petsc_arch = linux-gnu
   ...

Finally, you can build this distribution by typing::

   $ python setup.py build



Installing
----------

After building, this distribution is ready for installation. For a
site install type::

   $ python setup.py install

or, in case you need root privileges::

   $ su -c 'python setup.py install'

This will install the ``petsc4py`` package in the standard location
``<prefix>/lib/pythonX.X/site-packages``.

You can also do a user-install type::

   $ python setup.py install --home=$HOME

This will install the ``slepc4py`` package in the standard location
``$HOME/lib/python`` (or perhaps ``$HOME/lib64/python``). This
location should be listed in the ``PYTHONPATH`` environmental
variable.
