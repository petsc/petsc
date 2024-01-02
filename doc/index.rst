===============
PETSc |version|
===============
PETSc, the Portable, Extensible Toolkit for Scientific Computation,
pronounced PET-see (`/ˈpɛt-siː/ <https://en.wikipedia.org/wiki/Help:IPA/English#Key>`__), is
for the scalable (parallel) solution of scientific
applications modeled by partial differential equations (PDEs). It has bindings for C, Fortran, and Python (via petsc4py).
PETSc also contains TAO, the Toolkit for Advanced Optimization, software library.
It supports MPI, and GPUs through
CUDA, HIP, Kokkos, or OpenCL, as well as hybrid MPI-GPU parallelism; it also supports the NEC-SX Tsubasa Vector Engine.
Immediately jump in and run PETSc code :any:`handson`.

PETSc is developed as :ref:`open-source <doc_license>`, :any:`requests <doc_creepycrawly>` and :any:`contributions <ch_contributing>` are welcome.

.. admonition:: News: PETSc is associated with `NumFOCUS <http://numfocus.org>`__, a 501(c)(3) nonprofit supporting open code and reproducible science,
                through which you can help support PETSc.

  .. image:: /images/community/numfocus.png
     :align: center

.. admonition:: News: PETSc 2024 Annual Meeting

  The :any:`2024 Annual Meeting <meetings>` will take place May 23, 24 in Cologne, Germany.

.. admonition:: News: Book on numerical methods using PETSc

  **PETSc for Partial Differential Equations: Numerical Solutions in C and Python**, by Ed Bueler.

  - `Book from SIAM Press <https://my.siam.org/Store/Product/viewproduct/?ProductId=32850137>`__
  - `Google Play E-book <https://play.google.com/store/books/details/Ed_Bueler_PETSc_for_Partial_Differential_Equations?id=tgMHEAAAQBAJ>`__

Main Topics
===========

.. toctree::
   :maxdepth: 1

   overview/index
   install/index
   tutorials/index
   manual/index
   manualpages/index
   petsc4py/index
   faq/index
   community/index
   developers/index
   miscellaneous/index

`PETSc/TAO Users Manual in PDF <manual/manual.pdf>`__

.. _doc_toolkits_use_petsc:

Toolkits/libraries that use PETSc
=================================

-  `ADflow <https://github.com/mdolab/adflow>`__ An open-source
   computational fluid dynamics solver for aerodynamic and
   multidisciplinary optimization
-  `BOUT++ <https://boutproject.github.io>`__ Plasma simulation
   in curvilinear coordinate systems
-  `Chaste <https://www.cs.ox.ac.uk/chaste/>`__ Cancer, Heart and
   Soft Tissue Environment
-  `code_aster <https://www.code-aster.org/V2/spip.php?rubrique2>`__
   open-source general purpose finite element code for solid and
   structural mechanics
-  `COOLFluiD <https://github.com/andrealani/COOLFluiD>`__ CFD,
   plasma and multi-physics simulation package
-  `DAFoam <https://dafoam.github.io>`__ Discrete adjoint solvers
   with `OpenFOAM <https://openfoam.com>`__ for aerodynamic
   optimization
-  `DEAL.II <https://www.dealii.org/>`__ C++ based finite element
   simulation package
-  `DUNE-FEM <https://dune-project.org/sphinx/content/sphinx/dune-fem/>`__ Python and C++ based finite element simulation package
-  `FEniCS <https://fenicsproject.org/>`__ Python based finite
   element simulation package
-  `Firedrake <https://www.firedrakeproject.org/>`__ Python based
   finite element simulation package
-  `Fluidity <https://fluidityproject.github.io/>`__ a finite
   element/volume fluids code
-  `FreeFEM <https://freefem.org/>`__ finite element PDE solver
   with embedded domain specific language
-  `hIPPYlib <https://hippylib.github.io>`__ `FEniCS <https://fenicsproject.org/>`__-based toolkit
   for solving deterministic and Bayesian inverse
   problems governed by PDEs
-  `libMesh <https://libmesh.github.io>`__ adaptive finite element
   library
-  `MFEM <https://mfem.org/>`__ lightweight, scalable C++ library
   for finite element methods
-  `MLSVM <https://github.com/esadr/mlsvm>`__, Multilevel Support
   Vector Machines with PETSc.
-  `MoFEM <http://mofem.eng.gla.ac.uk/mofem/html>`__, An open
   source, parallel finite element library
-  `MOOSE - Multiphysics Object-Oriented Simulation
   Environment <https://mooseframework.inl.gov/>`__ finite element
   framework, built on `libMesh <https://libmesh.github.io>`__.
-  `OOFEM <http://www.oofem.org>`__ object-oriented finite element
   library
-  `OpenCarp <https://opencarp.org/>`__ Cardiac electrophysiology simulator
-  `OpenFOAM <https://develop.openfoam.com/modules/external-solver>`__
   Available as an extension for linear solvers for OpenFOAM
-  `OpenFPM <http://http://openfpm.mpi-cbg.de/>`__ framework for particles and mesh simulation
-  `OpenFVM <http://openfvm.sourceforge.net/>`__ finite volume
   based CFD solver
-  `PermonSVM <http://permon.vsb.cz/permonsvm.htm>`__ support
   vector machines and
   `PermonQP <http://permon.vsb.cz/permonqp.htm>`__ quadratic
   programming
-  `PetIGA <https://bitbucket.org/dalcinl/petiga/>`__ A framework
   for high performance Isogeometric Analysis
-  `PHAML <https://math.nist.gov/phaml/>`__ The Parallel
   Hierarchical Adaptive MultiLevel Project
-  `preCICE <https://www.precice.org>`__ - A fully parallel
   coupling library for partitioned multi-physics simulations
-  `PyClaw <https://www.clawpack.org/pyclaw/>`__ A massively
   parallel, high order accurate, hyperbolic PDE solver
-  `SLEPc <https://slepc.upv.es/>`__ Scalable Library for
   Eigenvalue Problems


.. _doc_index_citing_petsc:

Citing PETSc
============

You can run PETSc programs with the option ``-citations`` to print appropriate citations for the software and algorithms being used in that program.

For general citations on PETSc please use the following:

.. literalinclude:: /petsc.bib
   :language: none
   :start-at: petsc-web-page
   :end-at: year
   :append: }

.. literalinclude:: /petsc.bib
   :language: none
   :start-at: petsc-user-ref
   :end-at: year
   :append: }

.. literalinclude:: /petsc.bib
   :language: none
   :start-at: petsc-efficient
   :end-at: year
   :append: }

For petsc4py usage please cite

.. literalinclude:: /petsc.bib
   :language: none
   :start-at: dalcinpazklercosimo2011
   :end-at: year
   :append: }

For PETSc usage on GPUs please cite

.. literalinclude:: /petsc.bib
   :language: none
   :start-at: mills2021
   :end-at: author
   :append: }

For ``PetscSF`` -- parallel communication in PETSc -- please cite

.. literalinclude:: /petsc.bib
   :language: none
   :start-at: petscsf2022
   :end-at: pages
   :append: }

If you use the ``TS`` component of PETSc please cite the following:

.. literalinclude:: petsc.bib
   :language: none
   :start-at: abhyankaretal2018
   :end-at: year
   :append: }

If you utilize the ``TS`` adjoint solver please cite

.. literalinclude:: /petsc.bib
   :language: none
   :start-at: zhang2022tsadjoint
   :end-at: year
   :append: }
