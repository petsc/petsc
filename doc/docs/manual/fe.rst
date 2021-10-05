.. _chapter_fe:

PetscFE: Finite Element Infrastructure in PETSc
-----------------------------------------------

This chapter introduces the ``PetscFE`` class, and related subclasses ``PetscSpace`` and ``PetscDualSpace``, which are used to represent finite element discretizations. It details there interaction with the ``DMPLEX`` class to assemble functions and operators over computational meshes, and produce optimal solvers by constructing multilevel iterations, for example using ``PCPATCH``. The idea behind these classes is not to encompass all of computational finite elements, but rather to establish an interface and infrastructure that will allow PETSc to leverage the excellent work done in packages such as Firedrake, FEniCS, LibMesh, and Deal.II.

Using Pointwise Functions to Specify Finite Element Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the paper about `Unified Residual Evaluation <https://arxiv.org/abs/1309.1204>`__, which explains the use of pointwise evaluation functions to describe weak forms.

Describing a particular finite element problem to PETSc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A finite element problem is presented to PETSc in a series of steps. This is both to facilitate automation, and to allow multiple entry points for user code and external packages since so much finite element software already exists. First, we tell the ``DM``, usually a ``DMPLEX`` or ``DMFOREST``, that we have a set of finite element fields which we intended to solve for in our problem, using

::

  DMAddField(dm, NULL, presDisc);
  DMAddField(dm, channelLabel, velDisc);

The second argument is a ``DMLabel`` object indicating the support of the field on the mesh, with NULL indicating the entire domain. Once we have a set of fields, we calls

::

  DMCreateDS(dm);

This divides the computational domain into subdomains, called *regions* in PETSc, each with a unique set of fields supported on it. These subdomain are identified by labels, and each one has a ``PetscDS`` object describing the discrete system (DS) on that subdomain. There are query functions to get the set of DS objects for the DM, but it is usually easiest to get the proper DS for a given cell using

::

  DMGetCellDS(dm, cell, &ds);

Each `PetscDS`` object has a set of fields, each with a ``PetscFE`` discretization. This allows it to calculate the size of the local discrete approximation, as well as allocate scratch space for all the associated computations. The final thing needed is specify the actual equations to be enforced on each region. The ``PetscDS`` contains a ``PetscWeakForm`` object that holds callback function pointers that define the equations. A simplified, top-level interface through ``PetscDS`` allows users to quickly define problems for a single region. For example, in `SNES Tutorial ex13 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex13.c.html>`__, we define the Poisson problem using

::

  DMLabel  label;
  PetscInt f = 0, id = 1;

  PetscDSSetResidual(ds, f, f0_trig_inhomogeneous_u, f1_u);
  PetscDSSetJacobian(ds, f, f, NULL, NULL, NULL, g3_uu);
  PetscDSSetExactSolution(ds, f, trig_inhomogeneous_u, user);
  DMGetLabel(dm, "marker", &label);
  DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, f, 0, NULL, (void (*)(void)) ex, NULL, user, NULL);

where the pointwise functions are

::

  static PetscErrorCode trig_inhomogeneous_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
  {
    PetscInt d;
    *u = 0.0;
    for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
    return 0;
  }

  static void f0_trig_inhomogeneous_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
  {
    PetscInt d;
    for (d = 0; d < dim; ++d) f0[0] += -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
  }

  static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
  {
    PetscInt d;
    for (d = 0; d < dim; ++d) f1[d] = u_x[d];
  }

  static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
  {
    PetscInt d;
    for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
  }

Notice that we set boundary conditions using ``DMAddBoundary``, which will be described later in this chapter. Also we set an exact solution for the field. This can be used to automatically calculate mesh convergence using the ``PetscConvEst`` object described later in this chapter.

For more complex cases with multiple regions, we need to use the ``PetscWeakForm`` interface directly. The weak form object allows you to set any number of functions for a given field, and also allows functions to be associated with particular subsets of the mesh using labels and label values. We can reproduce the above problem using the *SetIndex* variants which only set a single function at the specified index, rather than a list of functions. We use a NULL label and value, meaning that the entire domain is used.

::

  PetscInt f = 0, val = 0;

  PetscDSGetWeakForm(ds, &wf);
  PetscWeakFormSetIndexResidual(ds, NULL, val, f, 0, 0, f0_trig_inhomogeneous_u, 0, f1_u);
  PetscWeakFormSetIndexJacobian(ds, NULL, val, f, f, 0, 0, NULL, 0, NULL, 0, NULL, 0, g3_uu);

In `SNES Tutorial ex23 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex23.c.html>`__, we define the Poisson problem over the entire domain, but in the top half we also define a pressure. The entire problem can be specified as follows

::

  DMGetRegionNumDS(dm, 0, &label, NULL, &ds);
  PetscDSGetWeakForm(ds, &wf);
  PetscWeakFormSetIndexResidual(wf, label, 1, 0, 0, 0, f0_quad_u, 0, f1_u);
  PetscWeakFormSetIndexJacobian(wf, label, 1, 0, 0, 0, 0, NULL, 0, NULL, 0, NULL, 0, g3_uu);
  PetscDSSetExactSolution(ds, 0, quad_u, user);
  DMGetRegionNumDS(dm, 1, &label, NULL, &ds);
  PetscDSGetWeakForm(ds, &wf);
  PetscWeakFormSetIndexResidual(wf, label, 1, 0, 0, 0, f0_quad_u, 0, f1_u);
  PetscWeakFormSetIndexJacobian(wf, label, 1, 0, 0, 0, 0, NULL, 0, NULL, 0, NULL, 0, g3_uu);
  PetscWeakFormSetIndexResidual(wf, label, 1, 1, 0, 0, f0_quad_p, 0, NULL);
  PetscWeakFormSetIndexJacobian(wf, label, 1, 1, 1, 0, 0, g0_pp, 0, NULL, 0, NULL, 0, NULL);
  PetscDSSetExactSolution(ds, 0, quad_u, user);
  PetscDSSetExactSolution(ds, 1, quad_p, user);
  DMGetLabel(dm, "marker", &label);
  DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) quad_u, NULL, user, NULL);

In the `PyLith software <https://geodynamics.org/cig/software/pylith/>`__ we use this capability to combine bulk elasticity with a fault constitutive model integrated over the embedded manifolds corresponding to earthquake faults.

Assembling finite element residuals and Jacobians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the pointwise functions are set in each ``PetscDS``, mesh traversals can be automatically determined from the ``DMLabel`` and value specifications in the keys. This default traversal strategy can be activated by attaching the ``DM`` and default callbacks to a solver

::

  SNESSetDM(snes, dm);
  DMPlexSetSNESLocalFEM(dm, &user, &user, &user);

  TSSetDM(ts, dm);
  DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user);
  DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &user);
  DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user);
