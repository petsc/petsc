
#include <petscdt.h>
#include <petscdraw.h>
#include <petscviewer.h>
#include <petscksp.h>
#include <petscdmda.h>

/*
      Solves -Laplacian u = f,  u(-1) = u(1) = 0 with multiple spectral elements

      Uses DMDA to manage the parallelization of the elements

      This is not intended to be highly optimized in either memory usage or time, but strifes for simplicity.

*/

typedef struct {
  PetscInt  n;                /* number of nodes */
  PetscReal *nodes;           /* GLL nodes */
  PetscReal *weights;         /* GLL weights */
} PetscGLL;

PetscErrorCode ComputeSolution(DM da,PetscGLL *gll,Vec u)
{
  PetscInt       j,xs,xn;
  PetscScalar    *uu,*xx;
  PetscReal      xd;
  Vec            x;

  PetscFunctionBegin;
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xn,NULL,NULL));
  PetscCall(DMGetCoordinates(da,&x));
  PetscCall(DMDAVecGetArray(da,x,&xx));
  PetscCall(DMDAVecGetArray(da,u,&uu));
  /* loop over local nodes */
  for (j=xs; j<xs+xn; j++) {
    xd    = xx[j];
    uu[j] = (xd*xd - 1.0)*PetscCosReal(5.*PETSC_PI*xd);
  }
  PetscCall(DMDAVecRestoreArray(da,x,&xx));
  PetscCall(DMDAVecRestoreArray(da,u,&uu));
  PetscFunctionReturn(0);
}

/*
      Evaluates \integral_{-1}^{1} f*v_i  where v_i is the ith basis polynomial via the GLL nodes and weights, since the v_i
      basis function is zero at all nodes except the ith one the integral is simply the weight_i * f(node_i)
*/
PetscErrorCode ComputeRhs(DM da,PetscGLL *gll,Vec b)
{
  PetscInt       i,j,xs,xn,n = gll->n;
  PetscScalar    *bb,*xx;
  PetscReal      xd;
  Vec            blocal,xlocal;

  PetscFunctionBegin;
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xn,NULL,NULL));
  xs   = xs/(n-1);
  xn   = xn/(n-1);
  PetscCall(DMGetLocalVector(da,&blocal));
  PetscCall(VecZeroEntries(blocal));
  PetscCall(DMDAVecGetArray(da,blocal,&bb));
  PetscCall(DMGetCoordinatesLocal(da,&xlocal));
  PetscCall(DMDAVecGetArray(da,xlocal,&xx));
  /* loop over local spectral elements */
  for (j=xs; j<xs+xn; j++) {
    /* loop over GLL points in each element */
    for (i=0; i<n; i++) {
      xd              = xx[j*(n-1) + i];
      bb[j*(n-1) + i] += -gll->weights[i]*(-20.*PETSC_PI*xd*PetscSinReal(5.*PETSC_PI*xd) + (2. - (5.*PETSC_PI)*(5.*PETSC_PI)*(xd*xd - 1.))*PetscCosReal(5.*PETSC_PI*xd));
    }
  }
  PetscCall(DMDAVecRestoreArray(da,xlocal,&xx));
  PetscCall(DMDAVecRestoreArray(da,blocal,&bb));
  PetscCall(VecZeroEntries(b));
  PetscCall(DMLocalToGlobalBegin(da,blocal,ADD_VALUES,b));
  PetscCall(DMLocalToGlobalEnd(da,blocal,ADD_VALUES,b));
  PetscCall(DMRestoreLocalVector(da,&blocal));
  PetscFunctionReturn(0);
}

/*
     Run with -build_twosided allreduce -pc_type bjacobi -sub_pc_type lu -q 16 -ksp_rtol 1.e-34 (or 1.e-14 for double precision)

     -q <q> number of spectral elements to use
     -N <N> maximum number of GLL points per element

*/
int main(int argc,char **args)
{
  PetscGLL       gll;
  PetscInt       N = 80,n,q = 8,xs,xn,j,l;
  PetscReal      **A;
  Mat            K;
  KSP            ksp;
  PC             pc;
  Vec            x,b;
  PetscInt       *rows;
  PetscReal      norm,xc,yc,h;
  PetscScalar    *f;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscDrawAxis  axis;
  DM             da;
  PetscMPIInt    rank,size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,NULL,NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-q",&q,NULL));

  PetscCall(PetscDrawCreate(PETSC_COMM_WORLD,NULL,"Log(Error norm) vs Number of GLL points",0,0,500,500,&draw));
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCall(PetscDrawLGCreate(draw,1,&lg));
  PetscCall(PetscDrawLGSetUseMarkers(lg,PETSC_TRUE));
  PetscCall(PetscDrawLGGetAxis(lg,&axis));
  PetscCall(PetscDrawAxisSetLabels(axis,NULL,"Number of GLL points","Log(Error Norm)"));

  for (n=4; n<N; n+=2) {

    /*
       da contains the information about the parallel layout of the elements
    */
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,q*(n-1)+1,1,1,NULL,&da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMDAGetInfo(da,NULL,&q,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
    q = (q-1)/(n-1);  /* number of spectral elements */

    /*
       gll simply contains the GLL node and weight values
    */
    PetscCall(PetscMalloc2(n,&gll.nodes,n,&gll.weights));
    PetscCall(PetscDTGaussLobattoLegendreQuadrature(n,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,gll.nodes,gll.weights));
    gll.n = n;
    PetscCall(DMDASetGLLCoordinates(da,gll.n,gll.nodes));

    /*
       Creates the element stiffness matrix for the given gll
    */
    PetscCall(PetscGaussLobattoLegendreElementLaplacianCreate(gll.n,gll.nodes,gll.weights,&A));

    /*
      Scale the element stiffness and weights by the size of the element
    */
    h    = 2.0/q;
    for (j=0; j<n; j++) {
      gll.weights[j] *= .5*h;
      for (l=0; l<n; l++) {
        A[j][l] = 2.*A[j][l]/h;
      }
    }

    /*
        Create the global stiffness matrix and add the element stiffness for each local element
    */
    PetscCall(DMCreateMatrix(da,&K));
    PetscCall(MatSetOption(K,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xn,NULL,NULL));
    xs   = xs/(n-1);
    xn   = xn/(n-1);
    PetscCall(PetscMalloc1(n,&rows));
    /*
        loop over local elements
    */
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<n; l++) rows[l] = j*(n-1)+l;
      PetscCall(MatSetValues(K,n,rows,n,rows,&A[0][0],ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY));

    PetscCall(MatCreateVecs(K,&x,&b));
    PetscCall(ComputeRhs(da,&gll,b));

    /*
        Replace the first and last rows/columns of the matrix with the identity to obtain the zero Dirichlet boundary conditions
    */
    rows[0] = 0;
    rows[1] = q*(n-1);
    PetscCall(MatZeroRowsColumns(K,2,rows,1.0,x,b));
    PetscCall(PetscFree(rows));

    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetOperators(ksp,K,K));
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCLU));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp,b,x));

    /* compute the error to the continium problem */
    PetscCall(ComputeSolution(da,&gll,b));
    PetscCall(VecAXPY(x,-1.0,b));

    /* compute the L^2 norm of the error */
    PetscCall(VecGetArray(x,&f));
    PetscCall(PetscGaussLobattoLegendreIntegrate(gll.n,gll.nodes,gll.weights,f,&norm));
    PetscCall(VecRestoreArray(x,&f));
    norm = PetscSqrtReal(norm);
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"L^2 norm of the error %" PetscInt_FMT " %g\n",n,(double)norm));
    PetscCheck(n <= 10 || norm <= 1.e-8,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Slower convergence than expected");
    xc   = (PetscReal)n;
    yc   = PetscLog10Real(norm);
    PetscCall(PetscDrawLGAddPoint(lg,&xc,&yc));
    PetscCall(PetscDrawLGDraw(lg));

    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&K));
    PetscCall(PetscGaussLobattoLegendreElementLaplacianDestroy(gll.n,gll.nodes,gll.weights,&A));
    PetscCall(PetscFree2(gll.nodes,gll.weights));
    PetscCall(DMDestroy(&da));
  }
  PetscCall(PetscDrawLGDestroy(&lg));
  PetscCall(PetscDrawDestroy(&draw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
      requires: !complex

   test:
     requires: !single

   test:
     suffix: 2
     nsize: 2
     requires: superlu_dist

TEST*/
