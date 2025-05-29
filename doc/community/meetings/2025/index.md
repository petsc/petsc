---
orphan: true
---

(2025_meeting)=

# 2025 PETSc Annual Users Meeting and Tutorial

## Meeting location

May 20-21, 2025, 101 Davis Hall, University of Buffalo, NY, USA ([105 White Rd, Amherst, NY 14260](https://maps.app.goo.gl/B38RsNe41Zd93rvX7))

## Meeting times

- Monday, May 19 - Tutorial (tutorials begin at 9am)
- Tuesday, May 20 - Meeting (begin at 9am)
- Wednesday, May 21 - Meeting (ends around 5pm)

## Agenda

[comment]: # (Intro: Python, Linear/Nonlinear Solver, GPU)

[comment2]: # (Adv: Meshing, SNESVI, Optimization)

### Monday, May 19: Tutorial

| Time     | Title                                                                                                                        | Speaker                 |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
|  9:00 am | Introduction                                                                                                                | [Matt Knepley]          |
|  9:15 am | Tutorial I: Introductory PETSc                                                                                              |                         |
| 12:00 pm | **Lunch** for tutorial attendees and early arrivees                                                                          |                         |
| 1:30 pm  | Emergent flow asymmetries from the metachronal motion of the soft flexible paddles of the gossamer worm                     | [Alexander Hoover]      |
| 2:00 pm  | Tutorial II: Advanced PETSc                                                                                                 |                         |
| 5:00 pm  | End of first day                                                                                                             |                         |

### Tuesday, May 20: Scientific Program

| Time     | Title                                                                                                                        | Speaker                 |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
|  9:00 am | Meeting Introduction                                                                                                                             | [Matt Knepley]          |
|  9:05 am | A projection method for particle resampling                                                                                  | [Mark Adams]            |
|  9:30 am | Dense Broyden-Fletcher-Goldfarb-Shanno (BFGS)                                                                                | [Hansol Suh]            |
| 10:00 am | IBAMR: Immersed-Boundary Adaptive Mesh Refinement                                                                           | [David Wells]           |
| 10:30 am | TaoTerm                                                                                                                     | [Toby Isaac]            |
| 10:45 am | **Coffee Break**                                                                                                          |                         |
| 11:00 am | Multiple RHS multigrid for the lattice Dirac equation                                                                        | [Peter Boyle]           |
| 11:30 am | DMSwarmRT: Ray tracing with PETSc's particle management library DMSwarm                                                      | [Joseph Pusztay]        |
| 12:00 pm | Empire AI                                                                                                                   | [Matt Jones]            |
| 12:15 pm | **Lunch**                                                                                           |                         |
|  1:30 pm | Exploring Quantum Phases of Interacting Lattice Models via Exact Diagonalization                                            | [Cheng-Chien Chen]      |
|  2:00 pm | Cardiac Fluid Dynamics                                                                                                       | [Boyce Griffith]        |
|  2:30 pm | Application of CutFEM and SCIFEM to the modeling of coastal processes through vegetation                                                                                                      | [Chris Kees]            |
|  3:00 pm | PetscRegressor                                                                                                  | [Richard Mills]         |
|  3:15 pm | **Poster Session and Coffee Break**                                                                                         |                         |
|  4:30 pm | **End of Posters**                                                                                                          |                         |
|  4:45 pm | Leave on bus for dinner at Niagara Falls                                                                                    |                         |


### Wednesday, May 21: Scientific Program

| Time     | Title                                                                                                                        | Speaker                 |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
|  9:00 am | TBA                                                                                                                         | [Blaise Bourdin]        |
|  9:30 am | Automatic Generation of Matrix-Free Routines for PDE Solvers with Devito via PETSc                                           | [Zoe Leibowitz]         |
| 10:00 am | PetscFD: Simplifying PDE Solutions                                                                                                       | [David Salac]           |
| 10:30 am | Implications of nonlinear rheology for plate tectonics                                                                      | [Margarete Jadamec]     |
| 10:45 am | **Coffee Break**                                                                                                          |                         |
| 11:00 am | Proteus Toolkit                                                                                                         | [Darsh Nathawani]       |
| 11:30 am | Mesh Transformations                                                                                                 | [Matthew Knepley]       |
| 12:00 pm | GitWorkflows                                                                                                                 | [Satish Balay]          |
| 12:15 pm | **Lunch**                                                                                           |                         |
|  1:30 pm | pyop3: A DSL for Unstructured Mesh Stencil Calculations                                                                      | [Conor Ward]            |
|  2:00 pm | IMEX in PETSc                                                                                                               | [Hong Zhang]            |
|  2:15 pm | Early Experiences in Building AI Assistants for Improving the Productivity of PETSc Users and Developers                                                                                                               | [Junchao Zhang, Hong Zhang]         |
|  2:30 pm | **PETSc Roundtable**                                                                                |                         |
|  3:30 pm | **Coffee Break**                                                                                                          |                         |
|  3:45 pm | **PETSc Roundtable**                                                                                                       |                         |
| 4:45 pm  | Meeting Closes                                                                                      |                         |

## List of Abstracts

(alexander-hoover)=

:::{topic} **Emergent flow asymmetries from the metachronal motion of the soft flexible paddles of the gossamer worm**
**Alexander Hoover**

Cleveland State University

Metachronal waves are ubiquitous in propulsive and fluid transport systems across many different scales and morphologies in the biological world. Gossamer worms, or tomopterids, are a soft-bodied, holopelagic worm that use metachrony with their flexible, gelatinous parapodia to deftly navigate the midwater ocean column that they inhabit. In the following study, we develop a three-dimensional, fluid-structure interaction model, using the IBAMR and libmesh frameworks, of a tomopterid parapodium to explore the emergent metachronal waves formed from the interplay of passive body elasticity, active muscular tension, and hydrodynamic forces. After introducing our model, we examine the effects that varying material properties have on the stroke of an individual parapodium as well as the resulting fluid dynamics. We then explore the temporal dynamics when multiple parapodia are placed sequentially and how differences in the phase can alter the collective kinematics and resulting flow field. Finally, we examine the role of phase differences in a freely-swimming model.
:::

(mark-adams)=

:::{topic} **A projection method for particle resampling**
**Mark Adams**

Lawrence Berkeley National Laboratory

Particle discretizations of partial differential equations are advantageous for high-dimensional kinetic models in phase space due to their better scalability than continuum approaches with respect to dimension. Complex processes collectively referred to as particle noise hamper long time simulations with particle methods. One approach to address this problem is particle mesh adaptivity or remapping, known as particle resampling. This talk introduces a resampling method that projects particles to and from a (finite element) function space. The method is simple; using standard sparse linear algebra and finite element techniques, it can adapt to almost any set of new particle locations and preserves all moments up to the order of polynomial represented exactly by the continuum function space.

This work is motivated by the Vlasov-Maxwell-Landau model of magnetized plasmas with up to six dimensions, 3X in physical space and 3V in velocity space, and is developed in the context of a 1X + 1V Vlasov-Poisson model of Landau damping with logically regular particle and continuum phase space grids. Stable long time dynamics are demonstrated up to T=500 and reproducibility artifacts and data with stable dynamics up to T=1000 are publicly available.
:::

(hansol-suh)=

:::{topic} **Dense Broyden-Fletcher-Goldfarb-Shanno (BFGS)**
**Hansol Suh**

Argonne National Laboratory

We will present a new dense formulation of BFGS specialize for the Limited Memory-Variable Metric (KSPLMVM) linear solver in PETSc, and illustrate its use for optimization problems.
:::

(david-wells)=

:::{topic} **IBAMR: Immersed-Boundary Adaptive Mesh Refinement**
**David Wells**

University of North Carolina, Chapel Hill

IBAMR is a parallel implementation of the immersed boundary method and other relevant numerics, such as Navier-Stokes and multiphase flow solvers. This presentation showcases some applications built on IBAMR and describes how they are fundamentally powered by PETSc.
:::

(joseph-pusztay)=

:::{topic} **DMSwarmRT: Ray tracing with PETSc's particle management library DMSwarm**
**Joseph Pusztay**

University at Buffalo

In this talk I will present work with DMSwarm, PETSc's parallel particle management library, to construct a general purpose ray trace with applicability to ICF plasma. I will discuss underlying improvements to the DMSwarm API to better support device side computation of swarm operations to facilitate the ray trace, with initial scalability tests and results. Additionally, I will present and discuss light weight time stepping objects for device side computation of systems with large numbers of fields that may be stepped independently.
:::

(cheng-chien-chen)=

:::{topic} **Exploring Quantum Phases of Interacting Lattice Models via Exact Diagonalization**
**Cheng-Chien Chen**

University of Alabama at Birmingham

Fermionic particles cannot occupy the same quantum state due to the Pauli exclusion principle. Therefore, solving the quantum many-body Schrödinger equation for electrons on finite-size lattices is equivalent to solving a finite-dimensional eigenvalue problem, where the matrix dimension grows exponentially with the lattice size. Here, I will discuss the exact diagonalization technique for finding the low-energy eigenstates of interacting fermionic models on two-dimensional lattices. These interacting models are shown to host a variety of emergent quantum phases, such as superconductivity and antiferromagnetism. For a sparse matrix with 34 billion basis states, the underlying code based on PETSc/SLEPc achieves a strong scaling performance of 85% linear scaling on more than 100,000 CPUs. The presentation will conclude with a brief discussion of potential future research directions, including ultra-large-scale matrix diagonalization based on matrix-free algorithms and/or quantum circuit simulations.
:::

(boyce-griffith)=

:::{topic} **Cardiac Fluid Dynamics**
**Boyce Griffith**

University of North Carolina, Chapel Hill

Cardiac fluid dynamics fundamentally involves interactions between complex blood flows and the structural deformations of the muscular heart walls and the thin, flexible valve leaflets. I will initially focus on models of an in vitro pulse-duplicator system that is commonly used in the development and regulation of prosthetic heart valves. These models enable detailed comparisons between experimental data and computational model predictions but use highly simplified descriptions of cardiac anatomy and physiology. I will also present recent in vitro models, focusing on a new comprehensive model of the human heart. This heart model includes fully three-dimensional descriptions of all major cardiac structures along with biomechanics models that are parameterized using experimental tensile test data obtained exclusively from human tissue specimens. Simulation results demonstrate that the model generates physiological stroke volumes, pressure-volume loops, and valvular pressure-flow relationships, thereby illustrating is its potential for predicting cardiac function in both health and disease. Time permitting, I will end the talk by describing extensions of this model to incorporate a comprehensive description of cardiac electrophysiology and electro-mechanical coupling.
:::

(zoe-leibowitz)=

:::{topic} **Automatic Generation of Matrix-Free Routines for PDE Solvers with Devito via PETSc**
**Zoe Leibowitz**

Imperial College, London

Traditional numerical solvers are often optimized for specific hardware architectures, making their adaptation to new computing environments challenging. The rapid evolution of hardware increases the complexity of rewriting and re-optimizing these solvers. By combining domain-specific languages (DSLs) with automated code generation, the level of abstraction is raised, enabling the generation of high-performance code across diverse hardware architectures. Moreover, providing users with a high-level problem specification facilitates the development of complex PDE solvers in a form closer to continuous mathematics, reducing code complexity and maximizing reuse.

Devito, a DSL and compiler for finite-difference solvers, has been extended to integrate iterative solver functionality through an interface with PETSc, enabling the generation of solvers for various computational fluid dynamics (CFD) problems. As an industry-standard framework, Devito automates the generation of highly optimized explicit finite-difference kernels and stencil computations and has been extensively used in large-scale seismic inversion and medical imaging applications. The new developments introduce automatic generation of matrix-free routines in Devito, allowing interaction with PETSc’s suite of solvers. Key enhancements include support for iterative solvers, implicit time-stepping, coupled solvers, and matrix-free preconditioning. These features are fully integrated into Devito’s symbolic API while maintaining compatibility with staggered grids, subdomains, and custom stencils.

This work expands Devito’s capabilities, enabling it to address a broader range of high-performance computing challenges, including incompressible flow problems in CFD. The new framework is demonstrated through benchmark simulations, including the backward-facing step and flow around a cylinder.
:::

(matt-knepley)=

:::{topic} **Making Meshes with DMPlexTransform**
**Matt Knepley**

University at Buffalo

Computational meshes, as a way to partition space, form the basis of much of PDE simulation technology, for instance for the finite element and finite volume discretization methods. In complex simulations, we are often driven to modify an input mesh. For example, to refine, coarsen, extrude, change cell types, or filter it. This code can be voluminous, error-prone, spread over many special cases, and hard to understand and maintain by subsequent developers. We present a simple, table-driven paradigm for mesh transformation which can execute a large variety of transformations in a performant, parallel manner, along with experiments in the open source library PETSc which can be run by the reader.
:::

(tim-steinhoff)=

:::{topic} **Using PETSc in a Multi-application Environment**
**Tim Steinhoff**

Gesellschaft für Anlagen- und Reaktorsicherheit (GRS) gGmbH

In this talk we provide an overview of the use of PETSc in the context of the code family AC<sup>2</sup> which is developed and distributed by GRS. AC<sup>2</sup> consists of multiple codes and is used to simulate the behavior of nuclear reactors during operation, transients, design basis and beyond design basis accidents up to radioactive releases to the environment. Access to PETSc is controlled by the self-developed wrapper NuT (Numerical Toolkit). We present a brief rundown of historical developments introducing NuT and therefore PETSc to handle certain numerical subtasks in AC<sup>2</sup>. This is accompanied by a deeper look into our latest development and the challenges that come with it in order to support the time evolution of nuclide inventories in burnup and decay calculations.
:::

(conor-ward)=

:::{topic} **pyop3: A DSL for Unstructured Mesh Stencil Calculations**
**Conor Ward**

Imperial College, London

pyop3 is a new domain-specific language that automates the application of local computational kernels over a mesh, termed 'unstructured mesh stencil calculations’. Such operations are ubiquitous across simulation methods including the finite element method and finite volume method, as well as preconditioners, slope limiters, and more. Written in Python, pyop3 takes advantage of some novel abstractions for describing mesh data (think generalised `PetscSection`) to describe complex mesh loops in a concise way that is agnostic to the underlying data layout. Having described the computation to be performed, pyop3 then uses just-in-time compilation to generate high-performance C code (CUDA/HIP coming soon) and coordinates its execution in parallel using MPI.

pyop3 is built on top of PETSc, wrapping many of its data types, and the design of the new data layout abstractions are strongly influenced by DMPlex.

This talk will introduce some of the novel abstractions that enable pyop3’s functionality before giving some examples of the sorts of computations that are expressible and the resulting code that is generated.
:::

(david-salac)=

:::{topic} **PetscFD: Simplifying PDE Solutions**
**David Salac**

University at Buffalo

This talk will outline recent efforts to include finite difference operations in PETSc through the addition of PetscFD. We begin by formally exploring the concept of stencil composition, showing that resulting stencil will have an accuracy equal to the lower of the two stencils being composed. The basic outline of PetscFD is then provided, in addition to several high-level functions that return matrices for arbitrary derivatives. Finally, the usage of PetscFD is demonstrated via several canonical examples.
:::

(darsh-nathawani)=

:::{topic} **Proteus Toolkit**
**Darsh Nathawani**

Louisiana State University

Proteus is a python package to solve PDEs using traditional and state-of-the-art numerical models. Proteus uses several C, C++ and Fortran libraries either as an external package or a part of Proteus. PETSc is a vital part of the development of Proteus. The objective of this talk is to introduce Proteus, explain how to get it and use it, and some initial performance tests using the Poisson problem and provide comparison with PETSc. This scaling analysis is a crucial part for a guidance to better design efficient algorithms.
:::

(chris-kees)=

:::{topic} **Application of CutFEM and SCIFEM to the modeling of coastal processes through vegetation**
**Chris Kees**

Louisiana State University

Understanding the effects of sea level rise on coastal ecosystems involves complex solid materials, such as mixed sediments and vegetation. Physical flume and basin studies have long been used in coastal engineering to understand wave and current dynamics around such structures. Numerical flumes based on computational fluid dynamics and fluid-structure interaction have recently begun to augment physical models for design studies, particularly for engineered structures where established Arbitrary Lagrangian-Eulerian (ALE) methods based on boundary-conforming meshes and isoparametric or isogeoemtric finite element methods are effective. The rapid growth of lidar and photogrammetry techniques at large scales and computed tomography at small scales has introduced the possibility of constructing numerical experiments for the complex natural materials in coastal ecosystems. These methods tend to produce low-order geometric representations with uneven resolution, which are typically not appropriate for conforming mesh generation. To address this challenge, recent work extended an existing ALE method to include embedded solid dynamics using a piecewise linear CutFEM approach. The implementation is based on equivalent polynomials. The approach retains the convergence properties of the CutFEM method while having a simple implementation within the existing two phase RANS model, which has been used frequently for numerical flume studies. This presentation will consider application and performance of the method for two critical coastal processes: wave interaction with vegetation and sediment dynamics.
:::

## Organizing Committees

### Extramural Committee
- [Blaise Bourdin](https://math.mcmaster.ca/~bourdinb/)
- [Danny Finn](https://scholar.google.com/citations?user=l09jI6wAAAAJ&hl=en)
- [Toby Isaac](https://tisaac.gitlab.io/triquadtethex/)
- [Lois McInnes](https://wordpress.cels.anl.gov/curfman/)
- [Louis Moresi](https://www.moresi.info/)
- [Darsh Nathawani](https://darshnathawani.com/)
- [Barry Smith](https://barrysmith.github.io/)
- [Junchao Zhang](https://www.anl.gov/profile/junchao-zhang)

### Local Committee
- [Margarete Jadamec](https://geovizlab.geology.buffalo.edu/)
- [Matt Jones](https://www.buffalo.edu/ccr/about-us/people/staff/jones.html)
- [Matt Knepley](https://cse.buffalo.edu/~knepley/)
- [Joseph Pusztay](https://www.linkedin.com/in/joseph-pusztay-174183129/)
- [David Salac](https://engineering.buffalo.edu/mechanical-aerospace/people/faculty/d-salac.html)

## Sponsors
```{image} https://petsc.gitlab.io/annual-meetings/2025/Center-for-Computational-Research.png
:width: 400
```
```{image} https://petsc.gitlab.io/annual-meetings/2025/Institute-for-Artificial-Intelligence-and-Data-Science-color.png
:width: 400
```

## Questions and Meeting Discussion

For questions about the meeting contact <mailto:petsc2025@lists.mcs.anl.gov>.
Join the discussion about the meeting at [PETSc on Discord](https://discord.gg/Fqm8r6Gcyb), [2025 PETSc Annual Users Meeting channel](https://discord.com/channels/1119324534303109172/1298348560600924200).

## Code of Conduct

All meeting attendees are expected to follow the PETSc/NumFocus Code of Conduct. The local committee will serve as the code of conduct response team, https://numfocus.org/code-of-conduct#response-team. Should any concerns arise during the meeting, please contact any response team member.


## Registration

Please [register](https://ti.to/nf-projects/petsc-annual-meeting) to save your seat.
Fee: \$100, for breaks and lunches; free for students.

## Submit a presentation

[Submit an abstract](https://docs.google.com/forms/d/126KwzajoQvcqU_q7btNsYxFqbe7rJ_vASC-tejZfXDQ) to be included in the schedule.
We welcome talks from all perspectives, including

- contributions to PETSc
- use of PETSc in applications or libraries
- development of libraries and packages [called from PETSc](https://petsc.org/release/install/external_software/)
- just curious about using PETSc in applications

## Student Travel Support

We have funding to provide travel support for students attending the meeting without their own funding. To apply, check the
"Student Funding Support" ticket while registering for the meeting. Early registration will increase your chance of obtaining travel support.

## Suggested hotels

- Hotels Near UB North

  - [Motel 6 Amherst, NY](https://www.motel6.com/en/home/property/buffalo-amherst.html) 4400 Maple Rd, Amherst, NY 14226, (716) 834-2231
  - [Hampton Inn Buffalo - Amherst](https://www.hilton.com/en/hotels/bufcphx-hampton-buffalo-amherst/) 1601 Amherst Manor Dr, Amherst, NY 14221, (716) 559-7010
  - [Candlewood Suites Buffalo Amherst](https://www.ihg.com/candlewood/hotels/us/en/amherst/bufcw/hoteldetail?cm_mmc=GoogleMaps-_-CW-_-US-_-BUFCW) 20 Flint Rd, Amherst, NY 14226, (716) 688-2100
  - [DoubleTree by Hilton Hotel Buffalo-Amherst](https://www.hilton.com/en/hotels/buffldt-doubletree-buffalo-amherst/) 10 Flint Rd, Amherst, NY 14226, (716) 689-4414
  - [Comfort Inn University](https://www.choicehotels.com/new-york/amherst/comfort-inn-hotels/ny293?mc=llgoxxpx) 1 Flint Rd, Amherst, NY 14226, (716) 415-1132
  - [Fairfield Inn & Suites Buffalo Amherst/University](https://www.marriott.com/en-us/hotels/buffn-fairfield-inn-and-suites-buffalo-amherst-university/overview/?scid=f2ae0541-1279-4f24-b197-a979c79310b0) 3880 Rensch Rd, Amherst, NY 14228, (716) 204-8936
  - [Staybridge Suites Buffalo-Amherst by IHG](https://www.ihg.com/staybridge/hotels/us/en/amherst/bufrr/hoteldetail?cm_mmc=GoogleMaps-_-SB-_-US-_-BUFRR}) 1290 Sweet Home Rd, Amherst, NY 14228, (716) 276-8750


- Hotels in Downtown Buffalo

  - [Holiday In Express & Suites Buffalo Downtown-Medical Ctr by IHG](https://www.ihg.com/holidayinnexpress/hotels/us/en/buffalo/bufms/hoteldetail?cm_mmc=GoogleMaps-_-EX-_-US-_-BUFMS) 601 Main St, Buffalo, NY 14203, (716) 854-5500, Located near a subway station
  - [Hilton Garden Inn Buffalo Downtown](https://www.hilton.com/en/hotels/bufmsgi-hilton-garden-inn-buffalo-downtown/?SEO_id=GMB-AMER-GI-BUFMSGI&y_source=1_MjA4MTcyMy03MTUtbG9jYXRpb24ud2Vic2l0ZQ%3D%3D) 10 Lafayette Square, Buffalo, NY 14203, (716) 848-1000, Located near a subway station
  - [Hampton Inn & Suites Buffalo Downtown](https://www.hilton.com/en/hotels/bufdthx-hampton-suites-buffalo-downtown/?SEO_id=GMB-AMER-HX-BUFDTHX&y_source=1_MjA4MzA5Ny03MTUtbG9jYXRpb24ud2Vic2l0ZQ%3D%3D) 220 Delaware Ave, Buffalo, NY 14202, (716) 855-2223, Located near Chippewa St/Nightlife
  - [Embassy Suites by Hilton Buffalo](https://www.hilton.com/en/hotels/bufeses-embassy-suites-buffalo/?SEO_id=GMB-AMER-ES-BUFESES&y_source=1_MTEwOTkxNC03MTUtbG9jYXRpb24ud2Vic2l0ZQ%3D%3D) 200 Delaware Ave, Buffalo, NY 14202, (716) 842-1000, Located near Chippewa St/Nightlife
  - [Curtiss Hotel](https://curtisshotel.com/) 210 Franklin St, Buffalo, NY 14202, (716) 954-4900, Located near Chippewa St/Nightlife

