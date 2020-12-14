.. _doc_multi:

**************************************
Maintaining Your PETSc Installation(s)
**************************************

Sometimes it is useful to have multiple concurrently built versions of PETSc installed,
for example you may have a ``arch-darwin-opt`` and ``arch-darwin-debug``. This would allow
you to run a highly optimized performance version of your code and a debug version simply
by changing the ``$PETSC_ARCH`` environment variable.

Environmental Variables ``$PETSC_DIR`` And ``$PETSC_ARCH``
==========================================================

.. note::

   Applications completely providing their own makefiles do not need to use ``$PETSC_DIR``
   or ``$PETSC_ARCH``

   ``$PETSC_ARCH`` is only needed for in-place installations. For out-of-place
   installations see :ref:`documentation <doc_config_install>`.

``$PETSC_DIR`` and ``$PETSC_ARCH`` (in-place installs only) are used by the PETSc
makefiles to indicate which directory and configuration of PETSc to use when compiling
examples or application code. These variables can be set as environment variables or
specified on the command line:

- Specify environment variable for bash (can be specified in ``~/.bashrc`` or
  ``~/.bash_profile``):

  .. code-block:: console

     > export PETSC_DIR=/absolute/path/to/petsc
     > export PETSC_ARCH=linux-gnu-c-debug


- Specify environment variable for csh/tcsh (can be specified in ``~/.cshrc``):

  .. code-block:: console

     > setenv PETSC_DIR /absolute/path/to/petsc
     > setenv PETSC_ARCH linux-gnu-c-debug

- Specify variable on commandline (bash) to build an example in
  ``$PETSC_DIR/src/ts/tutorials``:

  .. code-block:: console

     > PETSC_ARCH=linux-gnu-c-debug make PETSC_DIR=/absolute/path/to/petsc ex1

``$PETSC_DIR`` should point to the location of the PETSc installation. For out-of-place
installations this is the ``--prefix`` location. For in-place installations it is the
directory where you ran ``configure``.

.. _doc_multi_confcache:

Configure Caching
=================

The first time you build PETSc, you must first run ``configure`` which (among other
things) serves to identify key characteristics about the computing environment:

- The type of operating system
- Available compilers, their locations, and versions
- Location of common system header-files
- Location of common system libraries
- Whether your system supports certain instruction types (e.g. AVX)
- And more...

While many things that ``configure`` must check are liable to change in between
consecutive ``configure`` invocations, some things are very unlikely -- if ever -- to
change. Hence ``configure`` can safely cache and reuse these values in subsequent
``configure`` runs, helping to speed up the lengthy process. A similar system is also used
when determining whether to rebuild various packages installed through PETSc.

.. note::

   While the caching system used by ``configure`` is very useful, it usually errs on
   the side of caution. If a change has occured in your system which could reasonably
   change how a package/program is compiled, ``configure`` will automatically rebuild this
   software for you.

   If you would like to enforce that ``configure`` does not use any cached values, you may
   use the ``--force`` option when configuring your PETSc installation:

   .. code-block:: console

      > ./configure --some-args --force

   Keep in mind however that this will only disable ``configure``'s cache, not any other
   caching mechanism potentially in use, such as `CCache <https://ccache.dev/>`__.

.. admonition:: Caution
   :class: yellow

   If one still suspects malfeasance due to ``configure`` caching, or some corruption has
   occured due to a faulty ``configure`` one may use the nuclear option
   ``--with-clean``. This will **permanently delete all build files, including installed
   packages** under ``$PETSC_DIR/$PETSC_ARCH`` (effectively a "clean slate") before
   runnning ``configure``. The only thing preserved during this process is the
   ``reconfigure`` script:

   .. code-block:: console

      > ./configure --many-args --with-clean

Reconfigure
===========

For the reasons listed :ref:`above <doc_multi_confcache>`, the automatically generated
``$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/reconfigure-$PETSC_ARCH.py`` (henceforth refered
to simply as ``reconfigure``) script is generated for you upon successfully finishing
``configure``. ``reconfigure`` is a short-hand way of repeating your original
``configure`` invocation with the same arguments. In addition, ``reconfigure`` will also
always explicitly define ``PETSC_ARCH`` within the ``configure`` arguments, so there is no
need to specificy which PETSc installation you wish to reconfigure.

For example running the following ``configure``:

.. code-block:: console

   > ./configure --download-mpich --download-fblaslapack --with-debugging=1

Will result in the following ``reconfigure``:

.. code-block:: python

   #!/usr/local/opt/python@3.9/bin/python3.9
   if __name__ == '__main__':
     import sys
     import os
     sys.path.insert(0, os.path.abspath('config'))
     import configure
     configure_options = [
       '--download-mpich',
       '--download-fblaslapack',
       '--with-debugging=1',
       'PETSC_ARCH=arch-darwin-c-debug',
     ]
     configure.petsc_configure(configure_options)

In order to rerun this ``configure`` with the same arguments simply do:

.. code-block:: console

   > $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/reconfigure-$PETSC_ARCH.py

.. Note::

   The ``reconfigure`` script also comes with one additional powerful tool, namely the
   ability to additively set new ``configure`` options, and also to change the values of
   previous ``configure`` options! This is particularly useful if one has a lot of
   :ref:`external packages <doc_externalsoftware>` installed through PETSc and would like
   to install another.

   One need only call ``reconfigure``, supplying any additional command-line arguments as
   if it were the regular ``configure``. Suppose one had an installation of PETSc with the following arguments (represented in the ``reconfigure`` script):

   .. code-block:: python

      #!/usr/local/opt/python@3.9/bin/python3.9
      if __name__ == '__main__':
        import sys
        import os
        sys.path.insert(0, os.path.abspath('config'))
        import configure
        configure_options = [
          '--download-mpich',
          '--download-fblaslapack',
          '--with-debugging=1',
          'PETSC_ARCH=arch-darwin-c-debug',
        ]
        configure.petsc_configure(configure_options)

   Then calling it with new arguments:

   .. code-block:: console

      > $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/reconfigure-$PETSC_ARCH.py --download-zlib

   Will install `zlib <https://zlib.net/>`__ into that ``reconfigure``'s home
   ``$PETSC_ARCH``.

While it is automatically done for you the first time you ``configure`` and build PETSc,
it is useful to symlink the ``reconfigure`` script for each ``$PETSC_ARCH`` that you
intend to rebuild often into your ``$PETSC_DIR``:

.. code-block:: console

   > ln -s $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/reconfigure-$PETSC_ARCH.py $PETSC_DIR/

Updating or Reinstalling PETSc
==============================

If you follow the master or release branches of PETSc you can update your libraries with:

.. code-block:: console

   > git pull
   > make libs

Most of the time this will work, if there are errors regarding compiling Fortran stubs you
need to also do:

.. code-block:: console

   > make allfortranstubs
