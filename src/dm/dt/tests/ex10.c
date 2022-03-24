static char help[] =
  "Tests implementation of PetscSpace_Sum by solving the Poisson equations using a PetscSpace_Poly and a PetscSpace_Sum and checking that \
  solutions agree up to machine precision.\n\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscfe.h>
#include <petscsnes.h>
/* We are solving the system of equations:
 * \vec{u} = -\grad{p}
 * \div{u} = f
 */

/* Exact solutions for linear velocity
   \vec{u} = \vec{x};
   p = -0.5*(\vec{x} \cdot \vec{x});
   */
static PetscErrorCode linear_u(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  PetscInt c;

  for (c = 0; c < Nc; ++c) u[c] = x[c];
  return 0;
}

static PetscErrorCode linear_p(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  PetscInt d;

  u[0] = 0.;
  for (d=0; d<dim; ++d) u[0] += -0.5*x[d]*x[d];
  return 0;
}

static PetscErrorCode linear_divu(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  u[0] = dim;
  return 0;
}

/* fx_v are the residual functions for the equation \vec{u} = \grad{p}. f0_v is the term <v,u>.*/
static void f0_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt i;

  for (i=0; i<dim; ++i) f0[i] = u[uOff[0] + i];
}

/* f1_v is the term <v,-\grad{p}> but we integrate by parts to get <\grad{v}, -p*I> */
static void f1_v(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f1[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) {
    PetscInt d;

    for (d=0; d<dim; ++d) f1[c*dim + d] = (c==d) ? -u[uOff[1]] : 0;
  }
}

/* Residual function for enforcing \div{u} = f. */
static void f0_q_linear(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar rhs,divu=0;
  PetscInt    i;

  (void)linear_divu(dim,t,x,dim,&rhs,NULL);
  for (i=0; i< dim; ++i) divu += u_x[uOff_x[0]+i*dim+i];
  f0[0] = divu-rhs;
}

/* Boundary residual. Dirichlet boundary for u means u_bdy=p*n */
static void f0_bd_u_linear(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscScalar pressure;
  PetscInt    d;

  (void)linear_p(dim,t,x,dim,&pressure,NULL);
  for (d=0; d<dim; ++d) f0[d] = pressure*n[d];
}

/* gx_yz are the jacobian functions obtained by taking the derivative of the y residual w.r.t z*/
static void g0_vu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g0[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) g0[c*dim + c] = 1.0;
}

static void g1_qu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g1[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) g1[c*dim + c] = 1.0;
}

static void g2_vp(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g2[])
{
  PetscInt c;

  for (c=0; c<dim; ++c) g2[c*dim + c] = -1.0;
}

typedef struct
{
  PetscInt dummy;
} UserCtx;

static PetscErrorCode CreateMesh(MPI_Comm comm,UserCtx *user,DM *mesh)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, mesh));
  CHKERRQ(DMSetType(*mesh, DMPLEX));
  CHKERRQ(DMSetFromOptions(*mesh));
  CHKERRQ(DMSetApplicationContext(*mesh,user));
  CHKERRQ(DMViewFromOptions(*mesh,NULL,"-dm_view"));
  PetscFunctionReturn(0);
}

/* Setup the system of equations that we wish to solve */
static PetscErrorCode SetupProblem(DM dm,UserCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  PetscWeakForm  wf;
  const PetscInt id = 1;
  PetscInt       bd;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(dm, &ds));
  /* All of these are independent of the user's choice of solution */
  CHKERRQ(PetscDSSetResidual(ds,0,f0_v,f1_v));
  CHKERRQ(PetscDSSetResidual(ds,1,f0_q_linear,NULL));
  CHKERRQ(PetscDSSetJacobian(ds,0,0,g0_vu,NULL,NULL,NULL));
  CHKERRQ(PetscDSSetJacobian(ds,0,1,NULL,NULL,g2_vp,NULL));
  CHKERRQ(PetscDSSetJacobian(ds,1,0,NULL,g1_qu,NULL,NULL));

  CHKERRQ(DMGetLabel(dm, "marker", &label));
  CHKERRQ(PetscDSAddBoundary(ds,DM_BC_NATURAL,"Boundary Integral",label,1,&id,0,0,NULL,(void (*)(void))NULL,NULL,user,&bd));
  CHKERRQ(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  CHKERRQ(PetscWeakFormSetIndexBdResidual(wf, label, 1, 0, 0, 0, f0_bd_u_linear, 0, NULL));

  CHKERRQ(PetscDSSetExactSolution(ds,0,linear_u,NULL));
  CHKERRQ(PetscDSSetExactSolution(ds,1,linear_divu,NULL));
  PetscFunctionReturn(0);
}

/* Create the finite element spaces we will use for this system */
static PetscErrorCode SetupDiscretization(DM mesh,DM mesh_sum,PetscErrorCode (*setup)(DM,UserCtx*),UserCtx *user)
{
  DM             cdm = mesh,cdm_sum = mesh_sum;
  PetscFE        u,divu,u_sum,divu_sum;
  PetscInt       dim;
  PetscBool      simplex;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(mesh, &dim));
  CHKERRQ(DMPlexIsSimplex(mesh, &simplex));
  /* Create FE objects and give them names so that options can be set from
   * command line. Each field gets 2 instances (i.e. velocity and velocity_sum)created twice so that we can compare between approaches. */
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject)mesh),dim,dim,simplex,"velocity_",-1,&u));
  CHKERRQ(PetscObjectSetName((PetscObject)u,"velocity"));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject)mesh_sum),dim,dim,simplex,"velocity_sum_",-1,&u_sum));
  CHKERRQ(PetscObjectSetName((PetscObject)u_sum,"velocity_sum"));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject)mesh),dim,1,simplex,"divu_",-1,&divu));
  CHKERRQ(PetscObjectSetName((PetscObject)divu,"divu"));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject)mesh_sum),dim,1,simplex,"divu_sum_",-1,&divu_sum));
  CHKERRQ(PetscObjectSetName((PetscObject)divu_sum,"divu_sum"));

  CHKERRQ(PetscFECopyQuadrature(u,divu));
  CHKERRQ(PetscFECopyQuadrature(u_sum,divu_sum));

  /* Associate the FE objects with the mesh and setup the system */
  CHKERRQ(DMSetField(mesh,0,NULL,(PetscObject)u));
  CHKERRQ(DMSetField(mesh,1,NULL,(PetscObject)divu));
  CHKERRQ(DMCreateDS(mesh));
  CHKERRQ((*setup)(mesh,user));

  CHKERRQ(DMSetField(mesh_sum,0,NULL,(PetscObject)u_sum));
  CHKERRQ(DMSetField(mesh_sum,1,NULL,(PetscObject)divu_sum));
  CHKERRQ(DMCreateDS(mesh_sum));
  CHKERRQ((*setup)(mesh_sum,user));

  while (cdm) {
    CHKERRQ(DMCopyDisc(mesh,cdm));
    CHKERRQ(DMGetCoarseDM(cdm,&cdm));
  }

  while (cdm_sum) {
    CHKERRQ(DMCopyDisc(mesh_sum,cdm_sum));
    CHKERRQ(DMGetCoarseDM(cdm_sum,&cdm_sum));
  }

  /* The Mesh now owns the fields, so we can destroy the FEs created in this
   * function */
  CHKERRQ(PetscFEDestroy(&u));
  CHKERRQ(PetscFEDestroy(&divu));
  CHKERRQ(PetscFEDestroy(&u_sum));
  CHKERRQ(PetscFEDestroy(&divu_sum));
  CHKERRQ(DMDestroy(&cdm));
  CHKERRQ(DMDestroy(&cdm_sum));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  UserCtx         user;
  DM              dm,dm_sum;
  SNES            snes,snes_sum;
  Vec             u,u_sum;
  PetscReal       errNorm;
  const PetscReal errTol = PETSC_SMALL;
  PetscErrorCode  ierr;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  /* Set up a snes for the standard approach, one space with 2 components */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD,&user,&dm));
  CHKERRQ(SNESSetDM(snes,dm));

  /* Set up a snes for the sum space approach, where each subspace of the sum space represents one component */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes_sum));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD,&user,&dm_sum));
  CHKERRQ(SNESSetDM(snes_sum,dm_sum));
  CHKERRQ(SetupDiscretization(dm,dm_sum,SetupProblem,&user));

  /* Set up and solve the system using standard approach. */
  CHKERRQ(DMCreateGlobalVector(dm,&u));
  CHKERRQ(VecSet(u,0.0));
  CHKERRQ(PetscObjectSetName((PetscObject)u,"solution"));
  CHKERRQ(DMPlexSetSNESLocalFEM(dm,&user,&user,&user));
  CHKERRQ(SNESSetFromOptions(snes));
  CHKERRQ(DMSNESCheckFromOptions(snes,u));
  CHKERRQ(SNESSolve(snes,NULL,u));
  CHKERRQ(SNESGetSolution(snes,&u));
  CHKERRQ(VecViewFromOptions(u,NULL,"-solution_view"));

  /* Set up and solve the sum space system */
  CHKERRQ(DMCreateGlobalVector(dm_sum,&u_sum));
  CHKERRQ(VecSet(u_sum,0.0));
  CHKERRQ(PetscObjectSetName((PetscObject)u_sum,"solution_sum"));
  CHKERRQ(DMPlexSetSNESLocalFEM(dm_sum,&user,&user,&user));
  CHKERRQ(SNESSetFromOptions(snes_sum));
  CHKERRQ(DMSNESCheckFromOptions(snes_sum,u_sum));
  CHKERRQ(SNESSolve(snes_sum,NULL,u_sum));
  CHKERRQ(SNESGetSolution(snes_sum,&u_sum));
  CHKERRQ(VecViewFromOptions(u_sum,NULL,"-solution_sum_view"));

  /* Check if standard solution and sum space solution match to machine precision */
  CHKERRQ(VecAXPY(u_sum,-1,u));
  CHKERRQ(VecNorm(u_sum,NORM_2,&errNorm));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Sum space provides the same solution as a regular space: %s",(errNorm < errTol) ? "true" : "false");CHKERRQ(
    ierr);

  /* Cleanup */
  CHKERRQ(VecDestroy(&u_sum));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(SNESDestroy(&snes_sum));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(DMDestroy(&dm_sum));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 2d_lagrange
    requires: triangle
    args: -velocity_petscspace_degree 1 \
      -velocity_petscspace_type poly \
      -velocity_petscspace_components 2\
      -velocity_petscdualspace_type lagrange \
      -divu_petscspace_degree 0 \
      -divu_petscspace_type poly \
      -divu_petscdualspace_lagrange_continuity false \
      -velocity_sum_petscfe_default_quadrature_order 1 \
      -velocity_sum_petscspace_degree 1 \
      -velocity_sum_petscspace_type sum \
      -velocity_sum_petscspace_variables 2 \
      -velocity_sum_petscspace_components 2 \
      -velocity_sum_petscspace_sum_spaces 2 \
      -velocity_sum_petscspace_sum_concatenate true \
      -velocity_sum_petscdualspace_type lagrange \
      -velocity_sum_sumcomp_0_petscspace_type poly \
      -velocity_sum_sumcomp_0_petscspace_degree 1 \
      -velocity_sum_sumcomp_0_petscspace_variables 2 \
      -velocity_sum_sumcomp_0_petscspace_components 1 \
      -velocity_sum_sumcomp_1_petscspace_type poly \
      -velocity_sum_sumcomp_1_petscspace_degree 1 \
      -velocity_sum_sumcomp_1_petscspace_variables 2 \
      -velocity_sum_sumcomp_1_petscspace_components 1 \
      -divu_sum_petscspace_degree 0 \
      -divu_sum_petscspace_type sum \
      -divu_sum_petscspace_variables 2 \
      -divu_sum_petscspace_components 1 \
      -divu_sum_petscspace_sum_spaces 1 \
      -divu_sum_petscspace_sum_concatenate true \
      -divu_sum_petscdualspace_lagrange_continuity false \
      -divu_sum_sumcomp_0_petscspace_type poly \
      -divu_sum_sumcomp_0_petscspace_degree 0 \
      -divu_sum_sumcomp_0_petscspace_variables 2 \
      -divu_sum_sumcomp_0_petscspace_components 1 \
      -dm_refine 0 \
      -snes_error_if_not_converged \
      -ksp_rtol 1e-10 \
      -ksp_error_if_not_converged \
      -pc_type fieldsplit\
      -pc_fieldsplit_type schur\
      -divu_sum_petscdualspace_lagrange_continuity false \
      -pc_fieldsplit_schur_precondition full
TEST*/
