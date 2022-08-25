.. _chapter_dmbase:

DM Basics
----------

The previous chapters have focused on the core numerical solvers in PETSc. However, numerical solvers without efficient ways
(in both human and machine time) of connecting them to the mathematical models and discretizations that people wish to build their simulations on,
will not get widely used. Thus PETSc provides a set of abstractions represented by the ``DM`` object to provide a powerful, comprehensive
mechanism for translating the problem specification of a model and its discretization to the language and API of solvers. Some of the model
classes `DM` currently supports are finite difference methods for PDEs on structured and staggered grids (``DMDA`` and ``DMSTAG`` -- :any:`chapter_stag`),
PDEs on unstructured
grids with finite element and finite volume methods (``DMPLEX`` -- :any:`chapter_unstructured`), PDEs on quad and octree-grids (``DMFOREST``), models on
networks (graphs) such
as the power grid or river networks (``DMNETWORK`` -- :any:`chapter_network`), and particle-in-cell simulations (``DMSWARM``).

In previous chapters, we have demonstrated some simple usage of ``DM`` to provide the input for the solvers. In this chapter, and those that follow,
we will dive deep into the capabilities of ``DM``.


It is possible to create a  ``DM`` with

.. code-block::

   DMCreate(MPI_Comm comm,DM *dm);
   DMSetType(DM dm, DMType type);

but more commonly, a ``DM`` is created with a type-specific constructor; the construction process for each type of ``DM`` is discussed
in the sections on each ``DMType``. This chapter focuses
on commonalities between all the ``DM`` so we assume the ``DM`` already exists and wish to work with it.

As discussed earlier, a `DM` can construct vectors and matrices appropriate for a model and discretization and provide the mapping between the
global and local vector representations.

.. code-block::

   DMCreateLocalVector(DM dm,Vec *l);
   DMCreateGlobalVector(DM dm,Vec *g);
   DMGlobalToLocal(dm,g,l,INSERT_VALUES);
   DMLocalToGlobal(dm,l,g,ADD_VALUES);
   DMCreateMatrix(dm,Mat *m);

The matrices produced may support ``MatSetValuesLocal()`` allowing one to work with the local numbering on each MPI rank. For `DMDA` one can also
use ``MatSetValuesStencil()`` and for ``DMSTAG`` with ``DMStagMatSetValuesStencil()``.


A given ``DM`` can be refined for certain ``DMType``\s with ``DMRefine()`` or coarsened with ``DMCoarsen()``.
Mappings between ``DM``\s may be obtained with routines such as ``DMCreateInterpolation()``, ``DMCreateRestriction()`` and ``DMCreateInjection()``.

One can attach a `DM` to a solver object with

.. code-block::

   KSPSetDM(KSP ksp,DM dm);
   SNESSetDM(SNES snes,DM dm);
   TSSetDM(TS ts,DM dm);

Once the ``DM`` is attached, the solver can utilize it to create and process much of the data that the solver needs to set up and implement its solve.
For example, with ``PCMG`` simply providing a ``DM`` can allow it to create all the data structures needed to run geometric multigrid on your problem.

`SNES Tutorial ex19 <../../src/snes/tutorials/ex19.c.html>`__ demonstrates how this may be done with ``DMDA``.
