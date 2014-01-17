static char help[] = "\
Solves the constant-coefficient 27-point 3D Heat equation with an   \n\
Implicit Runge-Kutta method using MatKAIJ.                          \n\
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
  2       4th-order, 2-stage Gauss method                           \n\
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
#include <petscdmda.h>

/* define the IRK methods available */
#define IRKGAUSS      "gauss"

typedef struct {
  PetscReal     a;              /* diffusion coefficient      */
  PetscReal     xmin,xmax;      /* domain bounds              */
  PetscInt      niter;          /* number of time iterations  */
  PetscReal     dt;             /* time step size             */
} UserContext;

static PetscErrorCode ExactSolution(UserContext*,DM,PetscReal,Vec);
static PetscErrorCode RKCreate_Gauss(PetscInt,PetscScalar**,PetscScalar**,PetscReal**);
static PetscErrorCode ComputeMatrix(DM,Mat,void*);
static PetscErrorCode CreateIdentityLike(Mat B,Mat *Identity);

#include <petsc/private/kernels/blockinvert.h>

int main(int argc, char **argv)
{
  PetscErrorCode    ierr;
  Vec               u,uex,rhs,z;
  UserContext       ctx;
  PetscInt          nstages,is,ie,matis,matie,*ix,*ix2;
  PetscInt          n,s,t,total_its,mg_levels;
  PetscScalar       *A,*B,*At,*b,*zvals;
  PetscReal         *c,err,time;
  Mat               Identity,J,TA,SC,R;
  KSP               ksp;
  PetscFunctionList IRKList = NULL;
  char              irktype[256] = IRKGAUSS;
  DM                dm;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscFunctionListAdd(&IRKList,IRKGAUSS,RKCreate_Gauss);CHKERRQ(ierr);

  /* default value */
  ctx.a       = 1.0;
  ctx.xmin    = 0.0;
  ctx.xmax    = 1.0;
  ctx.niter   = 0;
  ctx.dt      = 0.0;
  mg_levels    = 1;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"IRK options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-a","diffusion coefficient","<1.0>",ctx.a,&ctx.a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-niter","number of time steps","<0>",ctx.niter,&ctx.niter,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","time step size","<0.0>",ctx.dt,&ctx.dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-irk_type","IRK method family","",IRKList,irktype,irktype,sizeof(irktype),NULL);CHKERRQ(ierr);
  nstages = 2;
  ierr = PetscOptionsInt ("-irk_nstages","Number of stages in IRK method","",nstages,&nstages,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mg_levels","Number of multigrid levels","",mg_levels,&mg_levels,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,-9,-9,-9,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&u);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm,&uex);CHKERRQ(ierr);

  /* initial solution */
  ierr = ExactSolution(&ctx,dm,0.0,u);CHKERRQ(ierr);
  /* exact solution */
  ierr = ExactSolution(&ctx,dm,ctx.dt*ctx.niter,uex);CHKERRQ(ierr);

  {                             /* Create A,b,c */
    PetscErrorCode (*irkcreate)(PetscInt,PetscScalar**,PetscScalar**,PetscReal**);
    ierr = PetscFunctionListFind(IRKList,irktype,&irkcreate);CHKERRQ(ierr);
    ierr = (*irkcreate)(nstages,&A,&b,&c);CHKERRQ(ierr);
  }
  {                             /* Invert A */
    PetscInt *pivots;
    PetscScalar *work;
    ierr = PetscMalloc2(nstages,&pivots,nstages,&work);CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A(nstages,A,pivots,work,PETSC_FALSE,NULL);CHKERRQ(ierr);
    ierr = PetscFree2(pivots,work);CHKERRQ(ierr);
  }
  /* Scale (1/dt)*A^{-1} and (1/dt)*b */
  for (s=0; s<nstages*nstages; s++) A[s] *= 1.0/ctx.dt;
  for (s=0; s<nstages; s++) b[s] *= (-ctx.dt);

  /* Compute row sums At and identity B */
  ierr = PetscMalloc2(nstages,&At,PetscSqr(nstages),&B);CHKERRQ(ierr);
  for (s=0; s<nstages; s++) {
    At[s] = 0;
    for (t=0; t<nstages; t++) {
      At[s] += A[s+nstages*t];      /* Row sums of  */
      B[s+nstages*t] = 1.*(s == t); /* identity */
    }
  }

  /* allocate and calculate the (-J) matrix */
  ierr = DMCreateMatrix(dm,&J);CHKERRQ(ierr);
  ierr = ComputeMatrix(dm,J,&ctx);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(J,&matis,&matie);CHKERRQ(ierr);
  ierr = CreateIdentityLike(J,&Identity);CHKERRQ(ierr);

  /* Create the KAIJ matrix for solving the stages */
  ierr = MatCreateKAIJ(J,nstages,nstages,A,B,&TA);CHKERRQ(ierr);

  /* Create the KAIJ matrix for step completion */
  ierr = MatCreateKAIJ(J,1,nstages,NULL,b,&SC);CHKERRQ(ierr);

  /* Create the KAIJ matrix to create the R for solving the stages */
  ierr = MatCreateKAIJ(Identity,nstages,1,NULL,At,&R);CHKERRQ(ierr);

  /* Create and set options for KSP */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,TA,TA);CHKERRQ(ierr);
  if (mg_levels > 1) {
    PC pc,pc0;
    DM dmc;
    Mat J0,TA0,P,MP;
    KSP ksp0;
    ierr = DMCoarsen(dm,MPI_COMM_NULL,&dmc);CHKERRQ(ierr);
    ierr = DMCreateInterpolation(dmc,dm,&P,NULL);CHKERRQ(ierr);
    ierr = MatPtAP(J,P,MAT_INITIAL_MATRIX,2.0,&J0);CHKERRQ(ierr);
    ierr = MatCreateKAIJ(J0,nstages,nstages,A,B,&TA0);CHKERRQ(ierr);
    ierr = MatCreateMAIJ(P,nstages,&MP);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
    ierr = PCMGSetLevels(pc,2,NULL);CHKERRQ(ierr);
    ierr = PCMGSetInterpolation(pc,1,MP);CHKERRQ(ierr);
    ierr = PCMGGetSmoother(pc,0,&ksp0);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp0,TA0,TA0);CHKERRQ(ierr);
    ierr = KSPSetType(ksp0,KSPGMRES);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp0,&pc0);CHKERRQ(ierr);
    ierr = PCSetType(pc0,PCPBJACOBI);CHKERRQ(ierr);
    ierr = MatDestroy(&J0);CHKERRQ(ierr);
    ierr = MatDestroy(&TA0);CHKERRQ(ierr);
    ierr = MatDestroy(&MP);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    ierr = DMDestroy(&dmc);CHKERRQ(ierr);
  }

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Allocate work and right-hand-side vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&z);CHKERRQ(ierr);
  ierr = VecSetFromOptions(z);CHKERRQ(ierr);
  ierr = VecSetSizes(z,(matie-matis)*nstages,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecDuplicate(z,&rhs);

  ierr = VecGetOwnershipRange(u,&is,&ie);CHKERRQ(ierr);
  ierr = PetscMalloc3(nstages,&ix,nstages,&zvals,ie-is,&ix2);CHKERRQ(ierr);
  /* iterate in time */
  for (n=0,time=0.,total_its=0; n<ctx.niter; n++) {
    PetscInt its;

    /* compute and set the right hand side */
    ierr = MatMult(R,u,rhs);CHKERRQ(ierr);

    /* Solve the system */
    ierr = KSPSolve(ksp,rhs,z);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    total_its += its;

    /* Update the solution */
    ierr = MatMultAdd(SC,z,u,u);CHKERRQ(ierr);

    /* time step complete */
    time += ctx.dt;
  }
  ierr = PetscFree3(ix,ix2,zvals);CHKERRQ(ierr);

  /* Deallocate work and right-hand-side vectors */
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);

  /* Calculate error in final solution */
  ierr = VecAYPX(uex,-1.0,u);
  ierr = VecNorm(uex,NORM_2,&err);
  // err  = PetscSqrtReal(err*err/((PetscReal)ctx.imax));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 norm of the numerical error = %g (time=%g)\n",(double)err,(double)time);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of time steps: %D (%D Krylov iterations)\n",ctx.niter,total_its);CHKERRQ(ierr);

  /* Free up memory */
  ierr = KSPDestroy(&ksp);      CHKERRQ(ierr);
  ierr = MatDestroy(&TA);       CHKERRQ(ierr);
  ierr = MatDestroy(&SC);       CHKERRQ(ierr);
  ierr = MatDestroy(&R);        CHKERRQ(ierr);
  ierr = MatDestroy(&J);        CHKERRQ(ierr);
  ierr = MatDestroy(&Identity); CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree3(A,b,c);     CHKERRQ(ierr);
  ierr = PetscFree2(At,B);      CHKERRQ(ierr);
  ierr = VecDestroy(&uex);      CHKERRQ(ierr);
  ierr = VecDestroy(&u);        CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&IRKList);CHKERRQ(ierr);

  PetscFinalize();
  return(0);
}

static PetscErrorCode ExactSolution(UserContext *ctx,DM dm,PetscReal t,Vec U)
{
  PetscErrorCode  ierr;
  PetscInt        i,j,k;
  DMDALocalInfo   info;
  PetscScalar     ***u;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(dm,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,U,&u);CHKERRQ(ierr);
  for (k=info.zs; k<info.zs+info.zm; k++) {
    for (j=info.ys; j<info.ys+info.ym; j++) {
      for (i=info.xs; i<info.xs+info.xm; i++) {
        PetscReal hx = 1/(info.mx-1),hy = 1/(info.my-1),hz = 1/(info.mz-1),x = i*hx,y = j*hy,z = j*hz;
        PetscReal r2 = PetscSqr(x - 0.5) + PetscSqr(y - 0.5) + PetscSqr(z - 0.5);
        u[k][j][i] = PetscExpScalar(-8 * r2);
      }
    }
  }
  ierr = DMDAVecRestoreArray(dm,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Arrays should be freed with PetscFree3(A,b,c) */
static PetscErrorCode RKCreate_Gauss(PetscInt nstages,PetscScalar **gauss_A,PetscScalar **gauss_b,PetscReal **gauss_c)
{
  PetscErrorCode    ierr;
  PetscScalar       *A,*G0,*G1;
  PetscReal         *b,*c;
  PetscInt          i,j;
  Mat               G0mat,G1mat,Amat;

  PetscFunctionBegin;
  ierr = PetscMalloc3(PetscSqr(nstages),&A,nstages,gauss_b,nstages,&c);CHKERRQ(ierr);
  ierr = PetscMalloc3(nstages,&b,PetscSqr(nstages),&G0,PetscSqr(nstages),&G1);CHKERRQ(ierr);
  ierr = PetscDTGaussQuadrature(nstages,0.,1.,c,b);CHKERRQ(ierr);
  for (i=0; i<nstages; i++) (*gauss_b)[i] = b[i]; /* copy to possibly-complex array */

  /* A^T = G0^{-1} G1 */
  for (i=0; i<nstages; i++) {
    for (j=0; j<nstages; j++) {
      G0[i*nstages+j] = PetscPowRealInt(c[i],j);
      G1[i*nstages+j] = PetscPowRealInt(c[i],j+1)/(j+1);
    }
  }
  /* The arrays above are row-aligned, but we create dense matrices as the transpose */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G0,&G0mat);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G1,&G1mat);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,A,&Amat);CHKERRQ(ierr);
  ierr = MatLUFactor(G0mat,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatMatSolve(G0mat,G1mat,Amat);CHKERRQ(ierr);
  ierr = MatTranspose(Amat,MAT_REUSE_MATRIX,&Amat);CHKERRQ(ierr);

  ierr = MatDestroy(&G0mat);CHKERRQ(ierr);
  ierr = MatDestroy(&G1mat);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  ierr = PetscFree3(b,G0,G1);CHKERRQ(ierr);
  *gauss_A = A;
  *gauss_c = c;
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMatrix(DM da,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
  PetscScalar    v[7],Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy;
  MatStencil     row,col[7];

  PetscFunctionBeginUser;
  ierr    = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx      = 1.0 / (PetscReal)(mx-1); Hy = 1.0 / (PetscReal)(my-1); Hz = 1.0 / (PetscReal)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;
  ierr    = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);

  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row.i = i; row.j = j; row.k = k;
        if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
          v[0] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
          ierr = MatSetValuesStencil(B,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else {
          v[0] = -HxHydHz;col[0].i = i; col[0].j = j; col[0].k = k-1;
          v[1] = -HxHzdHy;col[1].i = i; col[1].j = j-1; col[1].k = k;
          v[2] = -HyHzdHx;col[2].i = i-1; col[2].j = j; col[2].k = k;
          v[3] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);col[3].i = row.i; col[3].j = row.j; col[3].k = row.k;
          v[4] = -HyHzdHx;col[4].i = i+1; col[4].j = j; col[4].k = k;
          v[5] = -HxHzdHy;col[5].i = i; col[5].j = j+1; col[5].k = k;
          v[6] = -HxHydHz;col[6].i = i; col[6].j = j; col[6].k = k+1;
          ierr = MatSetValuesStencil(B,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateIdentityLike(Mat B,Mat *Identity)
{
  PetscErrorCode ierr;
  PetscInt matis,matie,i;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(B,&matis,&matie);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)B),Identity);CHKERRQ(ierr);
  ierr = MatSetType(*Identity,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(*Identity,matie-matis,matie-matis,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetUp(*Identity);CHKERRQ(ierr);
  for (i=matis; i<matie; i++) {
    ierr= MatSetValue(*Identity,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*Identity,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (*Identity,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
