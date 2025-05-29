# PETSc

PETSc, the Portable, Extensible Toolkit for Scientific Computation,
pronounced PET-see ([/ˈpɛt-siː/](https://en.wikipedia.org/wiki/Help:IPA/English#Key)), is
for the scalable (parallel) solution of scientific
applications modeled by partial differential equations (PDEs). It has bindings for C, Fortran, and Python (via petsc4py).
PETSc also contains TAO, the Toolkit for Advanced Optimization, software library.
They support MPI, and GPUs through
CUDA, HIP, Kokkos, or OpenCL, as well as hybrid MPI-GPU parallelism; they also support the NEC-SX Tsubasa Vector Engine.
Immediately jump in and run PETSc and TAO code {any}`handson`.

## News

The exciting {any}`2025 PETSc Annual User Meeting<2025_meeting>` recently took place May 20-21, 2025 in Buffalo, New York, USA.
Follow the link for abstracts and talks.

[SIAM News article](https://www.siam.org/publications/siam-news/articles/opencarp-personalized-computational-model-of-the-heart-examines-cardiac-rhythm/)
on the PETSc-powered [OpenCarp](https://opencarp.org/) cardiac electrophysiology simulator.

PETSc is now on [BlueSky](https://bsky.app/profile/petsc.org).

## Book

> **PETSc for Partial Differential Equations: Numerical Solutions in C and Python**, by Ed Bueler.
>
> - [Book from SIAM Press](https://my.siam.org/Store/Product/viewproduct/?ProductId=32850137)
> - [Google Play E-book](https://play.google.com/store/books/details/Ed_Bueler_PETSc_for_Partial_Differential_Equations?id=tgMHEAAAQBAJ)

## Main Topics

```{toctree}
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
```

<a href="./manual/manual.pdf">PETSc/TAO Users Manual in PDF</a>

(doc_toolkits_use_petsc)=

## Toolkits/libraries that use PETSc

- [ADflow](https://github.com/mdolab/adflow) An open-source
  computational fluid dynamics solver for aerodynamic and
  multidisciplinary optimization
- [BOUT++](https://boutproject.github.io) Plasma simulation
  in curvilinear coordinate systems
- [Chaste](https://www.cs.ox.ac.uk/chaste/) Cancer, Heart and
  Soft Tissue Environment
- [code_aster](https://www.code-aster.org/V2/spip.php?rubrique2)
  open-source general purpose finite element code for solid and
  structural mechanics
- [code_saturne](https://www.code-saturne.org)
  open-source general purpose code for fluid dynamics
- [COOLFluiD](https://github.com/andrealani/COOLFluiD) CFD,
  plasma and multi-physics simulation package
- [DAFoam](https://dafoam.github.io) Discrete adjoint solvers
  with [OpenFOAM](https://openfoam.com) for aerodynamic
  optimization
- [DAMASK](https://damask-multiphysics.org) Unified multi-physics
  crystal plasticity simulation package
- [DEAL.II](https://www.dealii.org/) C++ based finite element
  simulation package
- [DUNE-FEM](https://dune-project.org/sphinx/content/sphinx/dune-fem/) Python and C++ based finite element simulation package
- [FEniCS](https://fenicsproject.org/) Python based finite
  element simulation package
- [Firedrake](https://www.firedrakeproject.org/) Python based
  finite element simulation package
- [Fluidity](https://fluidityproject.github.io/) a finite
  element/volume fluids code
- [FreeFEM](https://freefem.org/) finite element and boundary element PDE solver
  with embedded domain specific language
- [GetDP](https://www.getdp.info/) a General Environment for the Treatment of Discrete Problems
- [hIPPYlib](https://hippylib.github.io) [FEniCS](https://fenicsproject.org/)-based toolkit
  for solving deterministic and Bayesian inverse
  problems governed by PDEs
- [libMesh](https://libmesh.github.io) adaptive finite element
  library
- [MFEM](https://mfem.org/) lightweight, scalable C++ library
  for finite element methods
- [MLSVM](https://github.com/esadr/mlsvm), Multilevel Support
  Vector Machines with PETSc.
- [MoFEM](http://mofem.eng.gla.ac.uk/mofem/html), An open
  source, parallel finite element library
- [MOOSE - Multiphysics Object-Oriented Simulation
  Environment](https://mooseframework.inl.gov/) finite element
  framework, built on [libMesh](https://libmesh.github.io).
- [OOFEM](http://www.oofem.org) object-oriented finite element
  library
- [OpenCarp](https://opencarp.org/) Cardiac electrophysiology simulator
- [OpenFOAM](https://develop.openfoam.com/modules/external-solver)
  Available as an extension for linear solvers for OpenFOAM
- [OpenFPM](https://openfpm.mpi-cbg.de/) framework for particles and mesh simulation
- [OpenFVM](http://openfvm.sourceforge.net/) finite volume
  based CFD solver
- [PermonSVM](http://permon.vsb.cz/permonsvm.htm) support
  vector machines and
  [PermonQP](http://permon.vsb.cz/permonqp.htm) quadratic
  programming
- [PetIGA](https://github.com/dalcinl/PetIGA) A framework
  for high performance Isogeometric Analysis
- [PFLOTRAN](https://pflotran.org/) An open source, state-of-the-art
  code for massively parallel simulation of subsurface flow, reactive transport, geomechanics, and electrical resistivity tomography
- [PHAML](https://math.nist.gov/phaml/) The Parallel
  Hierarchical Adaptive MultiLevel Project
- [preCICE](https://www.precice.org) - A fully parallel
  coupling library for partitioned multi-physics simulations
- [PyClaw](https://www.clawpack.org/pyclaw/) A massively
  parallel, high order accurate, hyperbolic PDE solver
- [SLEPc](https://slepc.upv.es/) Scalable Library for
  Eigenvalue Problems

(doc_index_citing_petsc)=

## Citing PETSc

You can run PETSc programs with the option `-citations` to print appropriate citations for the software and algorithms being used in that program.

For general citations on PETSc please use the following:

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: petsc-web-page
```

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: petsc-user-ref
```

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: petsc-efficient
```

For petsc4py usage please cite

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: dalcinpazklercosimo2011
```

For PETSc usage on GPUs please cite

```{literalinclude} /petsc.bib
:append: '}'
:end-at: author
:language: none
:start-at: mills2021
```

For `PetscSF` -- parallel communication in PETSc -- please cite

```{literalinclude} /petsc.bib
:append: '}'
:end-at: pages
:language: none
:start-at: petscsf2022
```

If you use the `TS` component of PETSc please cite the following:

```{literalinclude} petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: abhyankaretal2018
```

If you utilize the `TS` adjoint solver please cite

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: zhang2022tsadjoint
```
