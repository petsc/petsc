static char help[] = "Solves the constant-coefficient 1D heat equation \n\
with an Implicit Runge-Kutta method using MatKAIJ.                  \n\
                                                                    \n\
    du      d^2 u                                                   \n\
    --  = a ----- ; 0 <= x <= 1;                                    \n\
    dt      dx^2                                                    \n\
                                                                    \n\
  with periodic boundary conditions                                 \n\
                                                                    \n\
2nd order central discretization in space:                          \n\
                                                                    \n\
   [ d^2 u ]     u_{i+1} - 2u_i + u_{i-1}                           \n\
   [ ----- ]  =  ------------------------                           \n\
   [ dx^2  ]i              h^2                                      \n\
                                                                    \n\
    i = grid index;    h = x_{i+1}-x_i (Uniform)                    \n\
    0 <= i < n         h = 1.0/n                                    \n\
                                                                    \n\
Thus,                                                               \n\
                                                                    \n\
   du                                                               \n\
   --  = Ju;  J = (a/h^2) tridiagonal(1,-2,1)_n                     \n\
   dt                                                               \n\
                                                                    \n\
Implicit Runge-Kutta method:                                        \n\
                                                                    \n\
  U^(k)   = u^n + dt \\sum_i a_{ki} JU^{i}                          \n\
  u^{n+1} = u^n + dt \\sum_i b_i JU^{i}                             \n\
                                                                    \n\
  i = 1,...,s (s -> number of stages)                               \n\
                                                                    \n\
At each time step, we solve                                         \n\
                                                                    \n\
 [  1                                  ]     1                      \n\
 [ -- I \\otimes A^{-1} - J \\otimes I ] U = -- u^n \\otimes A^{-1} \n\
 [ dt                                  ]     dt                     \n\
                                                                    \n\
  where A is the Butcher tableaux of the implicit                   \n\
  Runge-Kutta method,                                               \n\
                                                                    \n\
with MATKAIJ and KSP.                                               \n\
                                                                    \n\
Available IRK Methods:                                              \n\
  gauss       n-stage Gauss method                                  \n\
                                                                    \n";

/*T
  Concepts: MATKAIJ
  Concepts: MAT
  Concepts: KSP
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
  petscsys.h      - base PETSc routines
  petscvec.h      - vectors
  petscmat.h      - matrices
  petscis.h       - index sets
  petscviewer.h   - viewers
  petscpc.h       - preconditioners
*/
#include <petscksp.h>
#include <petscdt.h>

/* define the IRK methods available */
#define IRKGAUSS      "gauss"

typedef enum {
  PHYSICS_DIFFUSION,
  PHYSICS_ADVECTION
} PhysicsType;
const char *const PhysicsTypes[] = {"DIFFUSION","ADVECTION","PhysicsType","PHYSICS_",NULL};

typedef struct __context__ {
  PetscReal     a;              /* diffusion coefficient      */
  PetscReal     xmin,xmax;      /* domain bounds              */
  PetscInt      imax;           /* number of grid points      */
  PetscInt      niter;          /* number of time iterations  */
  PetscReal     dt;             /* time step size             */
  PhysicsType   physics_type;
} UserContext;

static PetscErrorCode ExactSolution(Vec,void*,PetscReal);
static PetscErrorCode RKCreate_Gauss(PetscInt,PetscScalar**,PetscScalar**,PetscReal**);
static PetscErrorCode Assemble_AdvDiff(MPI_Comm,UserContext*,Mat*);

#include <petsc/private/kernels/blockinvert.h>

int main(int argc, char **argv)
{
  PetscErrorCode    ierr;
  Vec               u,uex,rhs,z;
  UserContext       ctxt;
  PetscInt          nstages,is,ie,matis,matie,*ix,*ix2;
  PetscInt          n,i,s,t,total_its;
  PetscScalar       *A,*B,*At,*b,*zvals,one = 1.0;
  PetscReal         *c,err,time;
  Mat               Identity,J,TA,SC,R;
  KSP               ksp;
  PetscFunctionList IRKList = NULL;
  char              irktype[256] = IRKGAUSS;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscFunctionListAdd(&IRKList,IRKGAUSS,RKCreate_Gauss));

  /* default value */
  ctxt.a       = 1.0;
  ctxt.xmin    = 0.0;
  ctxt.xmax    = 1.0;
  ctxt.imax    = 20;
  ctxt.niter   = 0;
  ctxt.dt      = 0.0;
  ctxt.physics_type = PHYSICS_DIFFUSION;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"IRK options","");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsReal("-a","diffusion coefficient","<1.0>",ctxt.a,&ctxt.a,NULL));
  CHKERRQ(PetscOptionsInt ("-imax","grid size","<20>",ctxt.imax,&ctxt.imax,NULL));
  CHKERRQ(PetscOptionsReal("-xmin","xmin","<0.0>",ctxt.xmin,&ctxt.xmin,NULL));
  CHKERRQ(PetscOptionsReal("-xmax","xmax","<1.0>",ctxt.xmax,&ctxt.xmax,NULL));
  CHKERRQ(PetscOptionsInt ("-niter","number of time steps","<0>",ctxt.niter,&ctxt.niter,NULL));
  CHKERRQ(PetscOptionsReal("-dt","time step size","<0.0>",ctxt.dt,&ctxt.dt,NULL));
  CHKERRQ(PetscOptionsFList("-irk_type","IRK method family","",IRKList,irktype,irktype,sizeof(irktype),NULL));
  nstages = 2;
  CHKERRQ(PetscOptionsInt ("-irk_nstages","Number of stages in IRK method","",nstages,&nstages,NULL));
  CHKERRQ(PetscOptionsEnum("-physics_type","Type of process to discretize","",PhysicsTypes,(PetscEnum)ctxt.physics_type,(PetscEnum*)&ctxt.physics_type,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* allocate and initialize solution vector and exact solution */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,ctxt.imax));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u,&uex));
  /* initial solution */
  CHKERRQ(ExactSolution(u  ,&ctxt,0.0));
  /* exact   solution */
  CHKERRQ(ExactSolution(uex,&ctxt,ctxt.dt*ctxt.niter));

  {                             /* Create A,b,c */
    PetscErrorCode (*irkcreate)(PetscInt,PetscScalar**,PetscScalar**,PetscReal**);
    CHKERRQ(PetscFunctionListFind(IRKList,irktype,&irkcreate));
    CHKERRQ((*irkcreate)(nstages,&A,&b,&c));
  }
  {                             /* Invert A */
    /* PETSc does not provide a routine to calculate the inverse of a general matrix.
     * To get the inverse of A, we form a sequential BAIJ matrix from it, consisting of a single block with block size
     * equal to the dimension of A, and then use MatInvertBlockDiagonal(). */
    Mat               A_baij;
    PetscInt          idxm[1]={0},idxn[1]={0};
    const PetscScalar *A_inv;
    CHKERRQ(MatCreateSeqBAIJ(PETSC_COMM_SELF,nstages,nstages,nstages,1,NULL,&A_baij));
    CHKERRQ(MatSetOption(A_baij,MAT_ROW_ORIENTED,PETSC_FALSE));
    CHKERRQ(MatSetValuesBlocked(A_baij,1,idxm,1,idxn,A,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(A_baij,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A_baij,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatInvertBlockDiagonal(A_baij,&A_inv));
    CHKERRQ(PetscMemcpy(A,A_inv,nstages*nstages*sizeof(PetscScalar)));
    CHKERRQ(MatDestroy(&A_baij));
  }
  /* Scale (1/dt)*A^{-1} and (1/dt)*b */
  for (s=0; s<nstages*nstages; s++) A[s] *= 1.0/ctxt.dt;
  for (s=0; s<nstages; s++) b[s] *= (-ctxt.dt);

  /* Compute row sums At and identity B */
  CHKERRQ(PetscMalloc2(nstages,&At,PetscSqr(nstages),&B));
  for (s=0; s<nstages; s++) {
    At[s] = 0;
    for (t=0; t<nstages; t++) {
      At[s] += A[s+nstages*t];      /* Row sums of  */
      B[s+nstages*t] = 1.*(s == t); /* identity */
    }
  }

  /* allocate and calculate the (-J) matrix */
  switch (ctxt.physics_type) {
  case PHYSICS_ADVECTION:
  case PHYSICS_DIFFUSION:
    CHKERRQ(Assemble_AdvDiff(PETSC_COMM_WORLD,&ctxt,&J));
  }
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Identity));
  CHKERRQ(MatSetType(Identity,MATAIJ));
  CHKERRQ(MatGetOwnershipRange(J,&matis,&matie));
  CHKERRQ(MatSetSizes(Identity,matie-matis,matie-matis,ctxt.imax,ctxt.imax));
  CHKERRQ(MatSetUp(Identity));
  for (i=matis; i<matie; i++) {
    CHKERRQ(MatSetValues(Identity,1,&i,1,&i,&one,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(Identity,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd  (Identity,MAT_FINAL_ASSEMBLY));

  /* Create the KAIJ matrix for solving the stages */
  CHKERRQ(MatCreateKAIJ(J,nstages,nstages,A,B,&TA));

  /* Create the KAIJ matrix for step completion */
  CHKERRQ(MatCreateKAIJ(J,1,nstages,NULL,b,&SC));

  /* Create the KAIJ matrix to create the R for solving the stages */
  CHKERRQ(MatCreateKAIJ(Identity,nstages,1,NULL,At,&R));

  /* Create and set options for KSP */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,TA,TA));
  CHKERRQ(KSPSetFromOptions(ksp));

  /* Allocate work and right-hand-side vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&z));
  CHKERRQ(VecSetFromOptions(z));
  CHKERRQ(VecSetSizes(z,PETSC_DECIDE,ctxt.imax*nstages));
  CHKERRQ(VecDuplicate(z,&rhs));

  CHKERRQ(VecGetOwnershipRange(u,&is,&ie));
  CHKERRQ(PetscMalloc3(nstages,&ix,nstages,&zvals,ie-is,&ix2));
  /* iterate in time */
  for (n=0,time=0.,total_its=0; n<ctxt.niter; n++) {
    PetscInt its;

    /* compute and set the right hand side */
    CHKERRQ(MatMult(R,u,rhs));

    /* Solve the system */
    CHKERRQ(KSPSolve(ksp,rhs,z));
    CHKERRQ(KSPGetIterationNumber(ksp,&its));
    total_its += its;

    /* Update the solution */
    CHKERRQ(MatMultAdd(SC,z,u,u));

    /* time step complete */
    time += ctxt.dt;
  }
  PetscFree3(ix,ix2,zvals);

  /* Deallocate work and right-hand-side vectors */
  CHKERRQ(VecDestroy(&z));
  CHKERRQ(VecDestroy(&rhs));

  /* Calculate error in final solution */
  CHKERRQ(VecAYPX(uex,-1.0,u));
  CHKERRQ(VecNorm(uex,NORM_2,&err));
  err  = PetscSqrtReal(err*err/((PetscReal)ctxt.imax));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"L2 norm of the numerical error = %g (time=%g)\n",(double)err,(double)time));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of time steps: %D (%D Krylov iterations)\n",ctxt.niter,total_its));

  /* Free up memory */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&TA));
  CHKERRQ(MatDestroy(&SC));
  CHKERRQ(MatDestroy(&R));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(MatDestroy(&Identity));
  CHKERRQ(PetscFree3(A,b,c));
  CHKERRQ(PetscFree2(At,B));
  CHKERRQ(VecDestroy(&uex));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(PetscFunctionListDestroy(&IRKList));

  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode ExactSolution(Vec u,void *c,PetscReal t)
{
  UserContext     *ctxt = (UserContext*) c;
  PetscInt        i,is,ie;
  PetscScalar     *uarr;
  PetscReal       x,dx,a=ctxt->a,pi=PETSC_PI;

  PetscFunctionBegin;
  dx = (ctxt->xmax - ctxt->xmin)/((PetscReal) ctxt->imax);
  CHKERRQ(VecGetOwnershipRange(u,&is,&ie));
  CHKERRQ(VecGetArray(u,&uarr));
  for (i=is; i<ie; i++) {
    x          = i * dx;
    switch (ctxt->physics_type) {
    case PHYSICS_DIFFUSION:
      uarr[i-is] = PetscExpScalar(-4.0*pi*pi*a*t)*PetscSinScalar(2*pi*x);
      break;
    case PHYSICS_ADVECTION:
      uarr[i-is] = PetscSinScalar(2*pi*(x - a*t));
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for physics type %s",PhysicsTypes[ctxt->physics_type]);
    }
  }
  CHKERRQ(VecRestoreArray(u,&uarr));
  PetscFunctionReturn(0);
}

/* Arrays should be freed with PetscFree3(A,b,c) */
static PetscErrorCode RKCreate_Gauss(PetscInt nstages,PetscScalar **gauss_A,PetscScalar **gauss_b,PetscReal **gauss_c)
{
  PetscScalar       *A,*G0,*G1;
  PetscReal         *b,*c;
  PetscInt          i,j;
  Mat               G0mat,G1mat,Amat;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc3(PetscSqr(nstages),&A,nstages,gauss_b,nstages,&c));
  CHKERRQ(PetscMalloc3(nstages,&b,PetscSqr(nstages),&G0,PetscSqr(nstages),&G1));
  CHKERRQ(PetscDTGaussQuadrature(nstages,0.,1.,c,b));
  for (i=0; i<nstages; i++) (*gauss_b)[i] = b[i]; /* copy to possibly-complex array */

  /* A^T = G0^{-1} G1 */
  for (i=0; i<nstages; i++) {
    for (j=0; j<nstages; j++) {
      G0[i*nstages+j] = PetscPowRealInt(c[i],j);
      G1[i*nstages+j] = PetscPowRealInt(c[i],j+1)/(j+1);
    }
  }
  /* The arrays above are row-aligned, but we create dense matrices as the transpose */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G0,&G0mat));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G1,&G1mat));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,A,&Amat));
  CHKERRQ(MatLUFactor(G0mat,NULL,NULL,NULL));
  CHKERRQ(MatMatSolve(G0mat,G1mat,Amat));
  CHKERRQ(MatTranspose(Amat,MAT_INPLACE_MATRIX,&Amat));

  CHKERRQ(MatDestroy(&G0mat));
  CHKERRQ(MatDestroy(&G1mat));
  CHKERRQ(MatDestroy(&Amat));
  CHKERRQ(PetscFree3(b,G0,G1));
  *gauss_A = A;
  *gauss_c = c;
  PetscFunctionReturn(0);
}

static PetscErrorCode Assemble_AdvDiff(MPI_Comm comm,UserContext *user,Mat *J)
{
  PetscInt       matis,matie,i;
  PetscReal      dx,dx2;

  PetscFunctionBegin;
  dx = (user->xmax - user->xmin)/((PetscReal)user->imax); dx2 = dx*dx;
  CHKERRQ(MatCreate(comm,J));
  CHKERRQ(MatSetType(*J,MATAIJ));
  CHKERRQ(MatSetSizes(*J,PETSC_DECIDE,PETSC_DECIDE,user->imax,user->imax));
  CHKERRQ(MatSetUp(*J));
  CHKERRQ(MatGetOwnershipRange(*J,&matis,&matie));
  for (i=matis; i<matie; i++) {
    PetscScalar values[3];
    PetscInt    col[3];
    switch (user->physics_type) {
    case PHYSICS_DIFFUSION:
      values[0] = -user->a*1.0/dx2;
      values[1] = user->a*2.0/dx2;
      values[2] = -user->a*1.0/dx2;
      break;
    case PHYSICS_ADVECTION:
      values[0] = -user->a*.5/dx;
      values[1] = 0.;
      values[2] = user->a*.5/dx;
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for physics type %s",PhysicsTypes[user->physics_type]);
    }
    /* periodic boundaries */
    if (i == 0) {
      col[0] = user->imax-1;
      col[1] = i;
      col[2] = i+1;
    } else if (i == user->imax-1) {
      col[0] = i-1;
      col[1] = i;
      col[2] = 0;
    } else {
      col[0] = i-1;
      col[1] = i;
      col[2] = i+1;
    }
    CHKERRQ(MatSetValues(*J,1,&i,3,col,values,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd  (*J,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST
 testset:
   suffix: 1
   args: -a 0.1 -dt .125 -niter 5 -imax 40 -ksp_monitor_short -pc_type pbjacobi -irk_type gauss -irk_nstages 2
   test:
     args: -ksp_atol 1e-6
   test:
     requires: hpddm !single
     suffix: hpddm
     output_file: output/ex74_1.out
     args: -ksp_atol 1e-6 -ksp_type hpddm
   test:
     requires: hpddm
     suffix: hpddm_gcrodr
     output_file: output/ex74_1_hpddm.out
     args: -ksp_atol 1e-4 -ksp_view_final_residual -ksp_type hpddm -ksp_hpddm_type gcrodr -ksp_hpddm_recycle 2
 test:
   suffix: 2
   args: -a 0.1 -dt .125 -niter 5 -imax 40 -ksp_monitor_short -pc_type pbjacobi -ksp_atol 1e-6 -irk_type gauss -irk_nstages 4 -ksp_gmres_restart 100
 testset:
   suffix: 3
   requires: !single
   args: -a 1 -dt .33 -niter 3 -imax 40 -ksp_monitor_short -pc_type pbjacobi -ksp_atol 1e-6 -irk_type gauss -irk_nstages 4 -ksp_gmres_restart 100 -physics_type advection
   test:
     args:
   test:
     requires: hpddm
     suffix: hpddm
     output_file: output/ex74_3.out
     args: -ksp_type hpddm
   test:
     requires: hpddm
     suffix: hpddm_gcrodr
     output_file: output/ex74_3_hpddm.out
     args: -ksp_view_final_residual -ksp_type hpddm -ksp_hpddm_type gcrodr -ksp_hpddm_recycle 5

TEST*/
