---
orphan: true
---

(2026-feb-cass-petsc-bof)=

*2026 Consortium for the Advancement of Scientific Software ([CASS](https://cass.community/)) BoF Days -- the PETSc/TAO Session (Virtual via Zoom)*


# PETSc/TAO: Recent Advances, User Experiences, and Community Discussion

**Time:** Feb 11, 2026, 10:00am - 11:30am (Central Time, US and Canada)

**Registration:** <del>https://argonne.zoomgov.com/meeting/register/ay4bMcRgSZaZ-l7u9AzAzQ</del> (expired)

**Note:** All participants **must register** to receive their own Zoom link. The step is brief and requires only a name and email address. The meeting will not be recorded.

## Agenda:

| Time | Speaker | Affiliation | Title |
| :---- | :---- | :---- | :---- |
| 10:00 AM | Barry Smith | Argonne National Laboratory | Welcome and introduction |
| 10:05 AM | Vincent Robert | CEA Paris-Saclay and Sorbonne Université | [Solving saddle-point systems from contact mechanics in HPC context using PETSc](#talk-vincent-robert) (Slides not available) |
| 10:12 AM | Bahaâ Sidi | EDF R\&D, ENPC, and Sorbonne Université | [Scalable domain decomposition solvers for hybrid high-order methods in PETSc](#talk-bahaa-sidi) ([Slides][s_01])  |
| 10:19 AM | Barry Smith, Richard Tran Mills, Hansol Suh, Junchao Zhang | Argonne National Laboratory | [PETSc new features update](#talk-petsc-anl) ([Slides-1][s_02], [Slides-2][s_03], [Slides-3][s_04])   |
| 10:42 AM | Xiaodong Liu | Remcom Inc | [A PETSc-based vector finite element method code for solving the frequency-domain Maxwell equations](#talk-xiaodong-liu)  (Slides not available)|
| 10:49 AM | Jeremy L Thompson | University of Colorado Boulder | [Ratel \- using MI300A APUs with libCEED and PETSc](#talk-jeremy-thompson) ([Slides][s_06]) |
| 10:56 AM | Darsh Nathawani | Louisiana State University | [PETSc in Proteus](#talk-darsh-nathawani) ([Slides][s_07]) |
| 11:03 AM | All participants |  | Users take this opportunity to provide feedback and make feature requests |
| 11:08 AM | Barry Smith | Argonne National Laboratory | Open the PETSc/AI discussion |
| 11:10 AM | Mark Adams | Lawrence Berkeley National Laboratory | [Using AI code assistants for large projects with PETSc](#talk-mark-adams) ([Slides][s_08])|
| 11:17 AM | All participants |  | Open discussion about PETSc in the era of AI and large language models |
| 11:30 AM | BoF concludes |  |  |



## List of Abstracts:

(talk-vincent-robert)=

:::{topic} **Solving saddle-point systems from contact mechanics in HPC context using PETSc**

**Speaker:** Vincent Robert

**Affiliation:** CEA Paris-Saclay and Sorbonne Université

**Abstract:** The French Alternative Energies and Atomic Energy Commission (CEA), is one of the leading French public research organisations. MANTA is a new CEA open-source industrial-strength simulation code which is dedicated to the solving, in a high performance computing context, of partial differential equations mainly from mechanics of deformable structures and their interactions. MANTA relies on PETSc to solve all its linear systems. Using PETSc, we focus on developing numerical methods to solve saddle-point systems arising from large-scale 3D contact problems, with the goal to include them in MANTA. Our approach is based on a projection method which transforms the indefinite saddle-point system into a positive definite reduced system. We propose several preconditioning matrices, introduce a scaling factor that improves the conditioning of the preconditioned system, and fix several issues related to the parallel distribution of the matrices. Using the Conjugate Gradient method as the iterative linear solver with an Algebraic Multigrid Method for the preconditioning, the current numerical tests show great scalability potential of the method.
:::


(talk-bahaa-sidi)=

:::{topic} **Scalable domain decomposition solvers for hybrid high-order methods in PETSc**

**Speaker:** Bahaâ Sidi

**Affiliation:** EDF R\&D, ENPC, and Sorbonne Université

**Abstract:** This talk presents ongoing work carried out at EDF R&D on the coupling of hybrid high-order (HHO) discretizations with scalable linear solvers available in PETSc. The objective is to efficiently solve large-scale linear elasticity problems arising from HHO methods, which naturally lead to block-structured systems separating cell and face unknowns. We focus on Schur complement strategies based on PCFIELDSPLIT, where the cell block is eliminated locally and the resulting interface problem is solved using a combination of explicit Schur complements, Krylov methods, and domain decomposition preconditioners.
:::


(talk-petsc-anl)=

:::{topic} **PETSc new features update**

**Speaker:** Barry Smith, Richard Tran Mills, Hansol Suh, Junchao Zhang

**Affiliation:** Argonne National Laboratory

**Abstract:** PETSc developers at Argonne will give an update on new features in PETSc, including the new PETSc Fortran bindings, PetscRegressor, TaoTerm, PETSc/GPU backends, mixed-precision and out-of-core facility support in PETSc/MUMPS, and integration with OpenFOAM via petsc4foam, among others.
:::


(talk-xiaodong-liu)=

:::{topic} **A PETSc-based vector finite element method code for solving the frequency-domain Maxwell equations**

**Speaker:** Xiaodong Liu

**Affiliation:** Remcom Inc

**Abstract:** A PETSc-based vector finite element method (FEM) code is under active development at Remcom Inc. for solving the frequency-domain Maxwell equations. The current implementation supports key electromagnetic analysis capabilities, including lumped-port excitation, perfectly matched layer (PML) boundary conditions, far-field radiation pattern computation, and input impedance evaluation.

The code is built on PETSc’s DMPlex infrastructure to manage fully unstructured meshes, enabling scalable assembly and solution workflows. Direct solvers based on MUMPS have been validated on problems with up to 13 million complex-valued degrees of freedom, with scalability currently constrained by available computational resources.

Ongoing work is twofold. First, we are investigating iterative solvers, in particular BDDC-preconditioned FGMRES, where additional challenges arise from the vector-valued nature of the FEM discretization. Second, future development will target expanded electromagnetic modeling capabilities and improved solver robustness and scalability, with an emphasis on industrial applicability and alignment with emerging exascale computing platforms.
:::


(talk-jeremy-thompson)=

:::{topic} **Ratel - using MI300A APUs with libCEED and PETSc**

**Speaker:** Jeremy L Thompson

**Affiliation:** University of Colorado Boulder

**Abstract:** AMD's MI300A APUs have unified memory accessible by both the CPU and APU cores on the accelerator, eliminating the need for communication between device and host memory spaces. In this talk, we discuss our work with the solid mechanics library Ratel to use this unified memory model for FEM and material point method (MPM) problems. Ratel uses PETSc's mesh management and solvers while relying upon libCEED for the local action of the linear and non-linear operators in a matrix-free fashion. Ratel has successfully scaled simulations to tens of billions of material points with the MI300As.
:::


(talk-darsh-nathawani)=

:::{topic} **PETSc in Proteus**

**Speaker:** Darsh Nathawani

**Affiliation:** Louisiana State University

**Abstract:** Proteus is a mixed language toolkit that offers models for continuum mechanical processes described by PDEs. It provides CG and DG discretization, classic linear and nonlinear solvers, and time integrators. PETSc is primarily integrated to use Vec and Mat objects as well as KSP and PC to build solvers. Currently, we are expanding PETSc support by using DMPlex for our meshing framework.
:::


(talk-mark-adams)=

:::{topic} **Using AI code assistants for large projects with PETSc**

**Speaker:** Mark Adams

**Affiliation:** Lawrence Berkeley National Laboratory

**Abstract:** I will discuss my experience over the past 4 months using AI code assistants (eg, Claude, Gemini, GPT5), integrated with IDEs (eg, VS Code with Roo) for large projects using or developing in PETSc. I have used chatGPT for several months for data processing and very small coding projects. GPT does not seem as good as Claude and Gemini at scientific code, such as PETSc, but I still find it useful for very quick problems.


I will give a quick demonstration and discuss my experience developing a data assimilation tool in PETSc for shallow water equations—both new topics for me. I will provide tips on using these tools and give an idea of how well they currently support developing PETSc C codes.
:::


[s_01]: https://petsc.gitlab.io/annual-meetings/BoFs/2026-02/BahaaSidi.pdf
[s_02]: https://petsc.gitlab.io/annual-meetings/BoFs/2026-02/RichardMills.pdf
[s_03]: https://petsc.gitlab.io/annual-meetings/BoFs/2026-02/HansolSuh.pdf
[s_04]: https://petsc.gitlab.io/annual-meetings/BoFs/2026-02/JunchaoZhang.pdf

[s_06]: https://petsc.gitlab.io/annual-meetings/BoFs/2026-02/JeremyThompson.pdf
[s_07]: https://petsc.gitlab.io/annual-meetings/BoFs/2026-02/DarshNathawani.pdf
[s_08]: https://petsc.gitlab.io/annual-meetings/BoFs/2026-02/MarkAdams.pdf
