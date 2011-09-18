========================
README: PETSc for Python
========================

:Author:       Lisandro Dalcin
:Contact:      dalcinl@gmail.com
:Web Site:     http://petsc4py.googlecode.com/
:Organization: CIMEC <http://www.cimec.org.ar/>
:Address:      CCT CONICET, 3000 Santa Fe, Argentina

Thank you for downloading the *PETSc for Python* project archive. As
this is a work in progress, please check the `project website`_ for
updates.  This project should be considered experimental, APIs are
subject to change at any time.

.. _CIMEC:            http://www.cimec.org.ar/
.. _project website:  http://petsc4py.googlecode.com/


- To build and install this package, you must meet the following
  requirements.

  + PETSc_ 3.2, 3.1 or 3.0.0, built with *shared libraries* (i.e., by
    passing ``--with-shared-libraries`` option to PETSc ``configure``
    script; this is not strictly required, but **highly** recommended).

  + Python_ 2.4 to 2.7 and 3.1 to 3.2.

  + NumPy_ 1.0.1 and above.

.. _PETSc:  http://www.mcs.anl.gov/petsc/
.. _Python: http://www.python.org
.. _NumPy:  http://numpy.scipy.org


- This package uses standard `distutils`. For detailed instructions
  about requirements and the building/install process, read the file
  ``docs/srouce/install.rst``.
