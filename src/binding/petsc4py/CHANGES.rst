=========================
CHANGES: PETSc for Python
=========================

:Author:  Lisandro Dalcin
:Contact: dalcinl@gmail.com


Release 3.17.0
==============

- Update to PETSc 3.17 release.


Release 3.16.0
==============

- Update to PETSc 3.16 release.


Release 3.15.0
==============

- Update to PETSc 3.15 release.


Release 3.14.1
==============

- Fix build issues after changes in recent PETSc patch releases.
- Add various missing types and enums definitions.
- Update Cython build, require ``Cython >= 0.24``.


Release 3.14.0
==============

- Update to PETSc 3.14 release.


Release 3.13.0
==============

- Update to PETSc 3.13 release.


Release 3.12.0
==============

- Update to PETSc 3.12 release.


Release 3.11.0
==============

- Update to PETSc 3.11 release.


Release 3.10.1
==============

- Fix for removal of ``SNESTEST``.
- Fix ``Mat`` in-place divide.


Release 3.10.0
==============

- Update to PETSc 3.10 release.


Release 3.9.1
=============

- Add ``Mat.zeroRowsColumnsLocal()``.
- Add ``Mat.getISLocalMat()``.
- Add ``Mat.convertISToAIJ()``.


Release 3.9.0
=============

- Update to PETSc 3.9 release.


Release 3.8.0
=============

- Update to PETSc 3.8 release.


Release 3.7.0
=============

- Update to PETSc 3.7 release.


Release 3.6.0
=============

- Update to PETSc 3.6 release.


Release 3.5.1
=============

- Add ``Log.{begin|view|destroy}()``.
- Add ``Mat.SOR()`` and ``Mat.SORType``.
- Add ``DMPlex.createCoarsePointIS()``.
- Add ``LGMap.createSF()``.
- Add ``SNES.getVIInactiveSet()``.
- Add ``Vec.isaxpy()``.
- Add ``PC.setReusePreconditioner()``.
- Return correct type in ``DM.getCoordinateDM()``.
- Fix SWIG wrappers to handle 64bit ``PetscInt``.
- Fix linker flags for Python from Fink.


Release 3.5
===========

- Update to PETSc 3.5 release.


Release 3.4
===========

- Update to PETSc 3.4 release.

- Add support for ``DMComposite`` and ``DMPlex``.

- Change ``Mat.getSizes()`` to return ``((m,M),(n,N))``.


Release 3.3.1
=============

- Fix ``Options.getAll()`` mishandling values with negative numbers.

- Minor backward compatibility fix for PETSc 3.2 .

- Minor bugfix for TSPYTHON subtype.


Release 3.3
===========

- Update to PETSc 3.3 release.

- Change ``Vec.getLocalForm()`` to ``Vec.localForm()`` for use with
  context manager and add ``Vec.setMPIGhost()``.

- Add ``AO.createMemoryScalable()`` and ``LGMap.block()`` /
  ``LGMap.unblock()``

- Add ``Object.handle`` property (C pointer as a Python integer). Can
  be used with ``ctypes`` to pass a PETSc handle.

- Add ``Comm.tompi4py()`` to get a ``mpi4py`` communicator instance.


Release 1.2
===========

- Update to PETSc 3.2 release.

- Add new ``DM`` class , make ``DA`` inherit from ``DM``.

- Better support for inplace LU/ILU and Cholesky/ICC factorization and
  factor PC subtypes.

- Now the ``Mat``/``PC``/``KSP``/``SNES``/``TS`` Python subtypes are
  implemented with Cython.

- Better interaction between Python garbage collector and PETSc
  objects.

- Support for PEP 3118 and legacy Python's buffer interface.


Release 1.1.2
=============

This is a new-features and bug-fix release.

- Add support for copying and computing complements in ``IS``
  (``IS.copy()`` and ``IS.complement()``).

- Add support for coarsening in ``DA`` (``DA.coarsen()``).

- Support for shallow copy and deep copy operations (use ``copy.copy``
  and ``copy.deepcopy``). Deep copy is only supported for a bunch of
  types (``IS``, ``Scatter``, ``Vec``, ``Mat``)

- Support for ``pip install petsc4py`` to download and install PETSc.


Release 1.1.1
=============

This is a new-features and bug-fix release.

- Support for setting PETSC_COMM_WORLD before PETSc initialization.

- Support for coordinates, refinement and interpolation in DA. Many
  thanks to Blaise Bourdin.

- Workaround build failures when PETSc is built with *mpiuni*.

- Workaround GIL-related APIs for non-threaded Python builds.


Release 1.1
===========

- Update for API cleanups, changes, and new calls in PETSc 3.1 and
  some other missing features.

- Add support for Jed Brown's THETA an GL timestepper implementations.

- Fix the annoying issues related to Open MPI shared libraries
  dependencies and Python dynamic loading.

- Many minor bug-fixes. Many thanks to Ethan Coon, Dmitry Karpeev,
  Juha Jaykka, and Michele De Stefano.


Release 1.0.3
=============

This is a bug-fix release.

- Added a quick fix to solve build issues. The macro __SDIR__ is no
  longer passed to the compiler in the command line.


Release 1.0.2
=============

This is a new-features and bug-fix release.

- Now ``petsc4py`` works against core PETSc built with complex
  scalars.

- Added support for PETSc logging features like stages, classes and
  events. Stages and events support the context manager interface
  (``with`` statement).

- Documentation generated with Epydoc and Sphinx is now included in
  the release tarball.

- Removed enumeration-like classes from the ``petsc4py.PETSc`` module
  namespace. For example, now you have to use ``PETSc.KSP.Type``
  instead of ``PETSc.KSPType``.

- The ``PETSc.IS`` to ``numpy.ndarray`` conversion now works for
  stride and block index sets.

- Implemented a more robust import machinery for multi-arch
  ``petsc4py`` installations. Now a wrong value in the ``PETSC_ARCH``
  environmental variable emit a warning (instead of failing) at import
  time.

- The unittest-based testsuite now can run under ``nose`` with its
  default options.

- Removed the dependency on ``numpy.distutils``, just use core Python
  ``distutils``.


Release 1.0.1
=============

This is a bug-fix release. Compile Cython-generated C sources with
``-Wwrite-strings`` removed, as this flag (inherited from PETSc) made
GCC emit a lot of (harmless but annoying) warnings about conversion of
string literals to non-const char pointers.


Release 1.0.0
=============

This is the fist release of the all-new, Cython-based, implementation
of *PETSc for Python*.
