========================
README: PETSc for Python
========================

:Author:       Lisandro Dalcin
:Organization: CIMEC_
:Address:      PTLC, (3000) Santa Fe, Argentina
:Contact:      dalcinl@gmail.com
:Web Site:     http://petsc4py.googlecode.com/

Thank you for downloading the *PETSc for Python* project archive. As
this is a work in progress, please check the `project website`_ for
updates.  This project should be considered experimental, APIs are
subject to change at any time.

.. _CIMEC:            http://www.cimec.org.ar/
.. _project website:  http://petsc4py.googlecode.com/


- To build and install this package, you must meet the following
  requirements.

  + PETSc_ 2.3.2/2.3.3/dev, built with *shared libraries*.

  + Python_ 2.4/2.5/2.6.

  + NumPy_ 1.0.1 and above.

.. _PETSc:  http://www-unix.mcs.anl.gov/petsc/petsc-as/
.. _Python: http://www.python.org
.. _NumPy:  http://numpy.scipy.org


- This package uses standard `distutils`. For detailed instructions
  about requirements and the building/install process, read the file
  ``docs/install.txt``.


- The project documentation can be found in files ``docs/*.txt``.  It
  is written reStructuredText_ format. You can use Docutils_ to get
  HTML or LaTeX output. A basic ``Makefile`` is provided in ``docs/``
  directory. 
  
  + Try ``make html`` to obtain HTML output in ``docs/petsc4py.html``.

  + Try ``make pdf``  to obtain PDF output in ``docs/petsc4py.pdf``.

.. _Docutils:         http://docutils.sourceforge.net
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
