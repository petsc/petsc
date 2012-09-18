static char help[] = "Simple test for using advanced discretizations with DMDA\n\n\n";

/*
TODO for Mantle Convection:
  - Variable viscosity
  - Free-slip boundary condition on upper surface
  - Stress-free boundary condition on sides and bottom
  - Parse Citcom input
  - Visualize output
*/

#include <petscdmda.h>
#include <petscsnes.h>

/*------------------------------------------------------------------------------
  This code can be generated using 'bin/pythonscripts/PetscGenerateFEMQuadratureTensorProduct.py dim order dim 1 laplacian dim order 1 1 gradient src/snes/examples/tutorials/ex67.h'
 -----------------------------------------------------------------------------*/
#include "ex67.h"

#define NUM_FIELDS 2 /* C89 Sucks Sucks Sucks Sucks: Cannot use static const values for array sizes */
const PetscInt numFields     = 2;
const PetscInt numComponents = NUM_BASIS_COMPONENTS_0+NUM_BASIS_COMPONENTS_1;

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscInt      debug;             /* The debugging level */
  PetscMPIInt   rank;              /* The process rank */
  PetscMPIInt   numProcs;          /* The number of processes */
  PetscBool     showInitial, showResidual, showJacobian, showSolution;
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  PetscLogEvent residualEvent, jacobianEvent, integrateResCPUEvent, integrateJacCPUEvent, integrateJacActionCPUEvent;
  /* Element quadrature */
  PetscQuadrature q[NUM_FIELDS];
  /* GPU partitioning */
  PetscInt      numBatches;        /* The number of cell batches per kernel */
  PetscInt      numBlocks;         /* The number of concurrent blocks per kernel */
  /* Problem definition */
  void        (*f0Funcs[NUM_FIELDS])(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]); /* The f_0 functions f0_u(x,y,z), and f0_p(x,y,z) */
  void        (*f1Funcs[NUM_FIELDS])(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]); /* The f_1 functions f1_u(x,y,z), and f1_p(x,y,z) */
  void        (*g0Funcs[NUM_FIELDS*NUM_FIELDS])(PetscScalar u[], const PetscScalar gradU[], PetscScalar g0[]); /* The g_0 functions g0_uu(x,y,z), g0_up(x,y,z), g0_pu(x,y,z), and g0_pp(x,y,z) */
  void        (*g1Funcs[NUM_FIELDS*NUM_FIELDS])(PetscScalar u[], const PetscScalar gradU[], PetscScalar g1[]); /* The g_1 functions g1_uu(x,y,z), g1_up(x,y,z), g1_pu(x,y,z), and g1_pp(x,y,z) */
  void        (*g2Funcs[NUM_FIELDS*NUM_FIELDS])(PetscScalar u[], const PetscScalar gradU[], PetscScalar g2[]); /* The g_2 functions g2_uu(x,y,z), g2_up(x,y,z), g2_pu(x,y,z), and g2_pp(x,y,z) */
  void        (*g3Funcs[NUM_FIELDS*NUM_FIELDS])(PetscScalar u[], const PetscScalar gradU[], PetscScalar g3[]); /* The g_3 functions g3_uu(x,y,z), g3_up(x,y,z), g3_pu(x,y,z), and g3_pp(x,y,z) */
} AppCtx;

void f0_u(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]) {
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;
  PetscInt       comp;

  for (comp = 0; comp < Ncomp; ++comp) {
    f0[comp] = 3.0;
  }
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z}
   u[Ncomp]          = {p} */
void f1_u(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]) {
  const PetscInt dim   = SPATIAL_DIM_0;
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;
  PetscInt       comp, d;

  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      /* f1[comp*dim+d] = 0.5*(gradU[comp*dim+d] + gradU[d*dim+comp]); */
      f1[comp*dim+d] = gradU[comp*dim+d];
    }
    f1[comp*dim+comp] -= u[Ncomp];
  }
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z} */
void f0_p(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]) {
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  f0[0] = 0.0;
  for (d = 0; d < dim; ++d) {
    f0[0] += gradU[d*dim+d];
  }
}

void f1_p(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]) {
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  for (d = 0; d < dim; ++d) {
    f1[d] = 0.0;
  }
}

/* < q, \nabla\cdot v >
   NcompI = 1, NcompJ = dim */
void g1_pu(PetscScalar u[], const PetscScalar gradU[], PetscScalar g1[]) {
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  for (d = 0; d < dim; ++d) {
    g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
  }
}

/* -< \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
void g2_up(PetscScalar u[], const PetscScalar gradU[], PetscScalar g2[]) {
  const PetscInt dim = SPATIAL_DIM_0;
  PetscInt       d;

  for (d = 0; d < dim; ++d) {
    g2[d*dim+d] = -1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
  }
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(PetscScalar u[], const PetscScalar gradU[], PetscScalar g3[]) {
  const PetscInt dim   = SPATIAL_DIM_0;
  const PetscInt Ncomp = NUM_BASIS_COMPONENTS_0;
  PetscInt       compI, d;

  for (compI = 0; compI < Ncomp; ++compI) {
    for (d = 0; d < dim; ++d) {
      g3[((compI*Ncomp+compI)*dim+d)*dim+d] = 1.0;
    }
  }
}

#undef __FUNCT__
#define __FUNCT__ "VecChop"
PetscErrorCode VecChop(Vec v, PetscReal tol)
{
  PetscScalar   *a;
  PetscInt       n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) {
    if (PetscAbsScalar(a[i]) < tol) a[i] = 0.0;
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->numBatches      = 1;
  options->numBlocks       = 1;
  options->showResidual    = PETSC_FALSE;
  options->showResidual    = PETSC_FALSE;
  options->showJacobian    = PETSC_FALSE;
  options->showSolution    = PETSC_TRUE;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "DMDA Test Problem Options", "DMDA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex62.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex62.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-gpu_batches", "The number of cell batches per kernel", "ex62.c", options->numBatches, &options->numBatches, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-gpu_blocks", "The number of concurrent blocks per kernel", "ex62.c", options->numBlocks, &options->numBlocks, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_initial", "Output the initial guess for verification", "ex62.c", options->showInitial, &options->showInitial, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_residual", "Output the residual for verification", "ex62.c", options->showResidual, &options->showResidual, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_jacobian", "Output the Jacobian for verification", "ex62.c", options->showJacobian, &options->showJacobian, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-show_solution", "Output the solution for verification", "ex62.c", options->showSolution, &options->showSolution, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("Residual",            SNES_CLASSID, &options->residualEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IntegResBatchCPU",    SNES_CLASSID, &options->integrateResCPUEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IntegJacBatchCPU",    SNES_CLASSID, &options->integrateJacCPUEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IntegJacActBatchCPU", SNES_CLASSID, &options->integrateJacActionCPUEvent);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Jacobian",            SNES_CLASSID, &options->jacobianEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim = user->dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch(dim) {
  case 2:
    ierr = DMDACreate2d(comm, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR, -4, -4, PETSC_DECIDE, PETSC_DECIDE, 2, 1, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMDACreate3d(comm, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR, -4, -4, -4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 2, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, dm);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Could not create mesh for dimension %d", dim);
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMDASetFieldName(*dm, 0, "velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(*dm, 1, "pressure");CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupQuadrature"
PetscErrorCode SetupQuadrature(AppCtx *user) {
  PetscFunctionBegin;
  user->q[0].numQuadPoints = NUM_QUADRATURE_POINTS_0;
  user->q[0].quadPoints    = points_0;
  user->q[0].quadWeights   = weights_0;
  user->q[0].numBasisFuncs = NUM_BASIS_FUNCTIONS_0;
  user->q[0].numComponents = NUM_BASIS_COMPONENTS_0;
  user->q[0].basis         = Basis_0;
  user->q[0].basisDer      = BasisDerivatives_0;
  user->q[1].numQuadPoints = NUM_QUADRATURE_POINTS_1;
  user->q[1].quadPoints    = points_1;
  user->q[1].quadWeights   = weights_1;
  user->q[1].numBasisFuncs = NUM_BASIS_FUNCTIONS_1;
  user->q[1].numComponents = NUM_BASIS_COMPONENTS_1;
  user->q[1].basis         = Basis_1;
  user->q[1].basisDer      = BasisDerivatives_1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
/* This is now Q_1-P_0 */
PetscErrorCode SetupSection(DM dm, AppCtx *user) {
  PetscInt       dim             = user->dim;
  PetscInt       numComp[2]      = {dim, 1};
  PetscInt       numVertexDof[2] = {dim, 0};
  PetscInt       numCellDof[2]   = {0, 1};
  PetscInt       numFaceDof[6];
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (d = 0; d < dim; ++d) {
    numFaceDof[0*dim + d] = 0;
    numFaceDof[1*dim + d] = 0;
  }
  ierr = DMDACreateSection(dm, numComp, numVertexDof, numFaceDof, numCellDof);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupExactSolution"
PetscErrorCode SetupExactSolution(AppCtx *user) {
  PetscFunctionBegin;
  user->f0Funcs[0] = f0_u;
  user->f0Funcs[1] = f0_p;
  user->f1Funcs[0] = f1_u;
  user->f1Funcs[1] = f1_p;
  user->g0Funcs[0] = PETSC_NULL;
  user->g0Funcs[1] = PETSC_NULL;
  user->g0Funcs[2] = PETSC_NULL;
  user->g0Funcs[3] = PETSC_NULL;
  user->g1Funcs[0] = PETSC_NULL;
  user->g1Funcs[1] = PETSC_NULL;
  user->g1Funcs[2] = g1_pu;      /* < q, \nabla\cdot v > */
  user->g1Funcs[3] = PETSC_NULL;
  user->g2Funcs[0] = PETSC_NULL;
  user->g2Funcs[1] = g2_up;      /* < \nabla\cdot v, p > */
  user->g2Funcs[2] = PETSC_NULL;
  user->g2Funcs[3] = PETSC_NULL;
  user->g3Funcs[0] = g3_uu;      /* < \nabla v, \nabla u + {\nabla u}^T > */
  user->g3Funcs[1] = PETSC_NULL;
  user->g3Funcs[2] = PETSC_NULL;
  user->g3Funcs[3] = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IntegrateResidualBatchCPU"
PetscErrorCode IntegrateResidualBatchCPU(PetscInt Ne, PetscInt numFields, PetscInt field, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscQuadrature quad[], void (*f0_func)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]), void (*f1_func)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]), PetscScalar elemVec[], AppCtx *user) {
  const PetscInt debug   = user->debug;
  const PetscInt dim     = SPATIAL_DIM_0;
  PetscInt       cOffset = 0;
  PetscInt       eOffset = 0, e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->integrateResCPUEvent,0,0,0,0);CHKERRQ(ierr);
  for (e = 0; e < Ne; ++e) {
    const PetscReal  detJ = jacobianDeterminants[e];
    const PetscReal *invJ = &jacobianInverses[e*dim*dim];
    const PetscInt   Nq   = quad[field].numQuadPoints;
    PetscScalar      f0[NUM_QUADRATURE_POINTS_0*dim];
    PetscScalar      f1[NUM_QUADRATURE_POINTS_0*dim*dim];
    PetscInt         q, f;

    if (Nq > NUM_QUADRATURE_POINTS_0) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_LIB, "Number of quadrature points %d should be <= %d", Nq, NUM_QUADRATURE_POINTS_0);
    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
    }
    for (q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      PetscScalar      u[dim+1];
      PetscScalar      gradU[dim*(dim+1)];
      PetscInt         fOffset            = 0;
      PetscInt         dOffset            = cOffset;
      const PetscInt   Ncomp       = quad[field].numComponents;
      const PetscReal *quadWeights = quad[field].quadWeights;
      PetscInt         d, f, i;

      for (d = 0; d <= dim; ++d)        {u[d]     = 0.0;}
      for (d = 0; d < dim*(dim+1); ++d) {gradU[d] = 0.0;}
      for (f = 0; f < numFields; ++f) {
        const PetscInt   Nb       = quad[f].numBasisFuncs;
        const PetscInt   Ncomp    = quad[f].numComponents;
        const PetscReal *basis    = quad[f].basis;
        const PetscReal *basisDer = quad[f].basisDer;
        PetscInt         b, comp;

        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscScalar    realSpaceDer[dim];
            PetscInt       d, g;

            u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              gradU[(fOffset+comp)*dim+d] += coefficients[dOffset+cidx]*realSpaceDer[d];
            }
          }
        }
        if (debug > 1) {
          PetscInt d;
          for (comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, u[fOffset+comp]);CHKERRQ(ierr);
            for (d = 0; d < dim; ++d) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, gradU[(fOffset+comp)*dim+d]);CHKERRQ(ierr);
            }
          }
        }
        fOffset += Ncomp;
        dOffset += Nb*Ncomp;
      }

      f0_func(u, gradU, &f0[q*Ncomp]);
      for (i = 0; i < Ncomp; ++i) {
        f0[q*Ncomp+i] *= detJ*quadWeights[q];
      }
      f1_func(u, gradU, &f1[q*Ncomp*dim]);
      for (i = 0; i < Ncomp*dim; ++i) {
        f1[q*Ncomp*dim+i] *= detJ*quadWeights[q];
      }
      if (debug > 1) {
        PetscInt c,d;
        for (c = 0; c < Ncomp; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "    f0[%d]: %g\n", c, f0[q*Ncomp+c]);CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]_%c: %g\n", c, 'x'+d, f1[(q*Ncomp + c)*dim+d]);CHKERRQ(ierr);
          }
        }
      }
      if (q == Nq-1) {cOffset = dOffset;}
    }
    for (f = 0; f < numFields; ++f) {
      const PetscInt   Nq       = quad[f].numQuadPoints;
      const PetscInt   Nb       = quad[f].numBasisFuncs;
      const PetscInt   Ncomp    = quad[f].numComponents;
      const PetscReal *basis    = quad[f].basis;
      const PetscReal *basisDer = quad[f].basisDer;
      PetscInt         b, comp;

      if (f == field) {
      for (b = 0; b < Nb; ++b) {
        for (comp = 0; comp < Ncomp; ++comp) {
          const PetscInt cidx = b*Ncomp+comp;
          PetscInt       q;

          elemVec[eOffset+cidx] = 0.0;
          for (q = 0; q < Nq; ++q) {
            PetscScalar realSpaceDer[dim];
            PetscInt    d, g;

            elemVec[eOffset+cidx] += basis[q*Nb*Ncomp+cidx]*f0[q*Ncomp+comp];
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              elemVec[eOffset+cidx] += realSpaceDer[d]*f1[(q*Ncomp+comp)*dim+d];
            }
          }
        }
      }
      if (debug > 1) {
        PetscInt b, comp;

        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    elemVec[%d,%d]: %g\n", b, comp, elemVec[eOffset+b*Ncomp+comp]);CHKERRQ(ierr);
          }
        }
      }
      }
      eOffset += Nb*Ncomp;
    }
  }
  /* ierr = PetscLogFlops((((2+(2+2*dim)*dim)*Ncomp*Nb+(2+2)*dim*Ncomp)*Nq + (2+2*dim)*dim*Nq*Ncomp*Nb)*Ne);CHKERRQ(ierr); */
  ierr = PetscLogEventEnd(user->integrateResCPUEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/*
  FormFunctionLocal - Form the local residual F from the local input X

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. F  - Local output vector

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: FormJacobianLocal()
*/
PetscErrorCode FormFunctionLocal(DM dm, Vec X, Vec F, AppCtx *user)
{
  const PetscInt   debug = user->debug;
  const PetscInt   dim   = user->dim;
  PetscReal       *coords, *v0, *J, *invJ, *detJ;
  PetscScalar     *elemVec, *u;
  PetscInt         cellDof  = 0;
  PetscInt         maxQuad  = 0;
  PetscInt         jacSize  = dim*dim;
  PetscInt         numCells, cStart, cEnd, c, field, d;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->residualEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = DMDAGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  for (field = 0; field < numFields; ++field) {
    cellDof += user->q[field].numBasisFuncs*user->q[field].numComponents;
    maxQuad  = PetscMax(maxQuad, user->q[field].numQuadPoints);
  }
  for (d = 0; d < dim; ++d) {jacSize *= maxQuad;}
  ierr = PetscMalloc3(dim,PetscReal,&coords,dim,PetscReal,&v0,jacSize,PetscReal,&J);CHKERRQ(ierr);
  ierr = PetscMalloc4(numCells*cellDof,PetscScalar,&u,numCells*jacSize,PetscReal,&invJ,numCells*maxQuad,PetscReal,&detJ,numCells*cellDof,PetscScalar,&elemVec);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscScalar *x;
    PetscInt           i;

    ierr = DMDAComputeCellGeometry(dm, c, &user->q[0], v0, J, &invJ[c*jacSize], &detJ[c]);CHKERRQ(ierr);
    if (detJ[c] <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ[c], c);
    ierr = DMDAVecGetClosure(dm, PETSC_NULL, X, c, &x);CHKERRQ(ierr);

    for (i = 0; i < cellDof; ++i) {
      u[c*cellDof+i] = x[i];
    }
  }
  for (field = 0; field < numFields; ++field) {
    const PetscInt numQuadPoints = user->q[field].numQuadPoints;
    const PetscInt numBasisFuncs = user->q[field].numBasisFuncs;
    void (*f0)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f0[]) = user->f0Funcs[field];
    void (*f1)(PetscScalar u[], const PetscScalar gradU[], PetscScalar f1[]) = user->f1Funcs[field];
    /* Conforming batches */
    PetscInt blockSize  = numBasisFuncs*numQuadPoints;
    PetscInt numBlocks  = 1;
    PetscInt batchSize  = numBlocks * blockSize;
    PetscInt numBatches = user->numBatches;
    PetscInt numChunks  = numCells / (numBatches*batchSize);
    ierr = IntegrateResidualBatchCPU(numChunks*numBatches*batchSize, numFields, field, u, invJ, detJ, user->q, f0, f1, elemVec, user);CHKERRQ(ierr);
    /* Remainder */
    PetscInt numRemainder = numCells % (numBatches * batchSize);
    PetscInt offset       = numCells - numRemainder;
    ierr = IntegrateResidualBatchCPU(numRemainder, numFields, field, &u[offset*cellDof], &invJ[offset*dim*dim], &detJ[offset],
                                     user->q, f0, f1, &elemVec[offset*cellDof], user);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    if (debug) {ierr = DMPrintCellVector(c, "Residual", cellDof, &elemVec[c*cellDof]);CHKERRQ(ierr);}
    ierr = DMDAVecSetClosure(dm, PETSC_NULL, F, c, &elemVec[c*cellDof], ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree4(u,invJ,detJ,elemVec);CHKERRQ(ierr);
  ierr = PetscFree3(coords,v0,J);CHKERRQ(ierr);
  if (user->showResidual) {
    PetscInt p;

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Residual:\n");CHKERRQ(ierr);
    for (p = 0; p < user->numProcs; ++p) {
      if (p == user->rank) {
        Vec f;

        ierr = VecDuplicate(F, &f);CHKERRQ(ierr);
        ierr = VecCopy(F, f);CHKERRQ(ierr);
        ierr = VecChop(f, 1.0e-10);CHKERRQ(ierr);
        ierr = VecView(f, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        ierr = VecDestroy(&f);CHKERRQ(ierr);
      }
      ierr = PetscBarrier((PetscObject) dm);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(user->residualEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
/*
  FormJacobianLocal - Form the local portion of the Jacobian matrix J from the local input X.

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. Jac  - Jacobian matrix

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

.seealso: FormFunctionLocal()
*/
PetscErrorCode FormJacobianLocal(DM dm, Vec X, Mat Jac, Mat JacP, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->jacobianEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->jacobianEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;                 /* nonlinear solver */
  Mat            A,J;                  /* Jacobian,preconditioner matrix */
  Vec            u,r;                  /* solution, residual vectors */
  AppCtx         user;                 /* user-defined work context */
  PetscReal      error = 0.0;          /* L_2 error in the solution */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, user.dm);CHKERRQ(ierr);

  ierr = SetupExactSolution(&user);CHKERRQ(ierr);
  ierr = SetupQuadrature(&user);CHKERRQ(ierr);
  ierr = SetupSection(user.dm, &user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMCreateMatrix(user.dm, MATAIJ, &J);CHKERRQ(ierr);
  A    = J;

  ierr = DMSetLocalFunction(user.dm, (DMLocalFunction1) FormFunctionLocal);CHKERRQ(ierr);
  ierr = DMSetLocalJacobian(user.dm, (DMLocalJacobian1) FormJacobianLocal);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, SNESDMComputeFunction, &user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, J, SNESDMComputeJacobian, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  {
    PetscReal res;

    /* Check discretization error */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial guess\n");CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* ierr = ComputeError(u, &error, &user);CHKERRQ(ierr); */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);
    /* Check residual */
    ierr = SNESDMComputeFunction(snes, u, r, &user);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial Residual\n");CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecView(r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
