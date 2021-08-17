===============
PETSc |version|
===============
PETSc, the Portable, Extensible Toolkit for Scientific Computation,
pronounced PET-see (`/ˈpɛt-siː/ <https://en.wikipedia.org/wiki/Help:IPA/English#Key>`__), is a suite of
data structures and routines for the scalable (parallel) solution of scientific
applications modeled by partial differential equations. It supports MPI, and GPUs through
CUDA or OpenCL, as well as hybrid MPI-GPU parallelism. PETSc (sometimes called PETSc/TAO)
also contains the TAO, the Toolkit for Advanced Optimization, software library.

PETSc is developed as :ref:`open-source <doc_license>`, requests and contributions are welcome.

News
====
.. admonition:: News: New Book on PETSc

  **PETSc for Partial Differential Equations: Numerical Solutions in C and Python**, by Ed Bueler, is available.

    - `Book from SIAM Press <https://my.siam.org/Store/Product/viewproduct/?ProductId=32850137>`__
    - `Google Play E-book <https://play.google.com/store/books/details/Ed_Bueler_PETSc_for_Partial_Differential_Equations?id=tgMHEAAAQBAJ>`__

.. admonition:: News: New paper on PETSc communication System

  `The PetscSF Scalable Communication Layer <https://arxiv.org/abs/2102.13018>`__

.. admonition:: News: New paper on PETSc with GPUs

  `Toward Performance-Portable PETSc for GPU-based Exascale Systems <https://arxiv.org/abs/2011.00715>`__

.. admonition:: News: petsc4py Support

  Source code for `petsc4py <https://www.mcs.anl.gov/petsc/petsc4py-current/docs/>`__, developed by Lisandro Dalcin, is now distributed with the PETSc source and supported by the PETSc team and mailing lists.


Main Topics
===========

.. toctree::
   :maxdepth: 1

   Overview <overview/index>
   Download <download/index>
   Installation <install/index>
   FAQ <faq/index>
   Documentation <docs/index>
   Tutorials <tutorials/guide_to_examples>
   Community <community/index>
   Developers <developers/index>
   Misc. <miscellaneous/index>

Related toolkits/libraries that use PETSc
==========================================

      -  `ADflow <https://github.com/mdolab/adflow>`__ An Open-Source
         Computational Fluid Dynamics Solver for Aerodynamic and
         Multidisciplinary Optimization
      -  `BOUT++ <https://boutproject.github.io>`__ Plasma simulation
         in curvilinear coordinate systems
      -  `Chaste <https://www.cs.ox.ac.uk/chaste/>`__ Cancer, Heart and
         Soft Tissue Environment
      -  `code_aster <https://www.code-aster.org/V2/spip.php?rubrique2​>`__
         open source general purpose finite element code for solid and
         structural mechanics
      -  `COOLFluiD <https://github.com/andrealani/COOLFluiD>`__ CFD,
         plasma and multi-physics simulation package
      -  `DAFoam <https://dafoam.github.io>`__ Discrete adjoint solvers
         with `OpenFOAM <https://openfoam.com>`__ for aerodynamic
         optimization
      -  `DEAL.II <https://www.dealii.org/>`__ C++ based finite element
         simulation package
      -  `FEniCS <https://fenicsproject.org/>`__ Python based finite
         element simulation package
      -  `Firedrake <https://www.firedrakeproject.org/>`__ Python based
         finite element simulation package
      -  `Fluidity <https://fluidityproject.github.io/>`__ a finite
         element/volume fluids code
      -  `FreeFEM <https://freefem.org/>`__ finite element PDE solver
         with embedded domain specific language
      -  `hIPPYlib <https://hippylib.github.io>`__ FEniCS based toolkit
         for solving large-scale deterministic and Bayesian inverse
         problems governed by partial differential equations
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
         framework, built on top of libMesh and PETSc
      -  `OOFEM <http://www.oofem.org>`__ object oriented finite element
         library
      -  `OpenCarp <https://opencarp.org/>`__ Cardiac Electrophysiology Simulator
      -  `OpenFOAM <https://develop.openfoam.com/modules/external-solver>`__
         Available as an extension for linear solvers for OpenFOAM
      -  `OpenFVM <http://openfvm.sourceforge.net/>`__ finite volume
         based CFD solver
      -  `PermonSVM <http://permon.it4i.cz/permonsvm.htm>`__ support
         vector machines and
         `PermonQP <http://permon.it4i.cz/permonqp.htm>`__ quadratic
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

When citing PETSc in a publication please cite the following:

.. code-block:: none

   @Misc{petsc-web-page,
      Author = "Satish Balay and Shrirang Abhyankar and Mark~F. Adams and Jed Brown
      and Peter Brune and Kris Buschelman and Lisandro Dalcin and Alp Dener and Victor Eijkhout
      and William~D. Gropp and Dinesh Kaushik and Matthew~G. Knepley and Dave~A. May
      and Lois Curfman McInnes and Richard Tran Mills and Todd Munson and Karl Rupp
      and Patrick Sanan and Barry~F. Smith and Stefano Zampini and Hong Zhang and Hong Zhang",
      Title  = "{PETS}c {W}eb page",
      Note   = "https://petsc.org/",
      Year   = "2021"}

   @TechReport{petsc-user-ref,
      Author = "Satish Balay and Shrirang Abhyankar and Mark~F. Adams and Jed Brown
      and Peter Brune and Kris Buschelman and Lisandro Dalcin and Alp Dener and Victor Eijkhout
      and William~D. Gropp and Dinesh Kaushik and Matthew~G. Knepley and Dave~A. May
      and Lois Curfman McInnes and Richard Tran Mills and Todd Munson and Karl Rupp
      and Patrick Sanan and Barry~F. Smith and Stefano Zampini and Hong Zhang and Hong Zhang",
      Title       = "{PETS}c Users Manual",
      Number      = "ANL-95/11 - Revision 3.15",
      Institution = "Argonne National Laboratory",
      Year        = "2021"}

   @InProceedings{petsc-efficient,
      Author    = "Satish Balay and William D. Gropp and Lois C. McInnes and Barry F. Smith",
      Title     = "Efficient Management of Parallelism in Object Oriented
                   Numerical Software Libraries",
      Booktitle = "Modern Software Tools in Scientific Computing",
      Editor    = "E. Arge and A. M. Bruaset and H. P. Langtangen",
      Pages     = "163--202",
      Publisher = "Birkhauser Press",
      Year      = "1997"}

If you use the TS component of PETSc please cite the following:

.. code-block:: none

   @article{abhyankar2018petsc,
     title={PETSc/TS: A Modern Scalable ODE/DAE Solver Library},
     author={Abhyankar, Shrirang and Brown, Jed and Constantinescu, Emil M and Ghosh, Debojyoti and Smith, Barry F and Zhang, Hong},
     journal={arXiv preprint arXiv:1806.01437},
     year={2018}
   }

If you utilize the TS adjoint solver please cite

.. code-block:: none

     @article{zhang2019,
       author = {Zhang, Hong and Constantinescu, Emil M. and Smith, Barry F.},
       title = {{{PETSc TSAdjoint: a discrete adjoint ODE solver for first-order and second-order sensitivity analysis}},
       journal = {arXiv e-preprints},
       eprint = {1912.07696},
       archivePrefix = {arXiv},
       year={2019}
     }
