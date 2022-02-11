
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
  PetscErrorCode ierr;
  PetscInt       j,xs,xn;
  PetscScalar    *uu,*xx;
  PetscReal      xd;
  Vec            x;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,x,&xx);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,u,&uu);CHKERRQ(ierr);
  /* loop over local nodes */
  for (j=xs; j<xs+xn; j++) {
    xd    = xx[j];
    uu[j] = (xd*xd - 1.0)*PetscCosReal(5.*PETSC_PI*xd);
  }
  ierr = DMDAVecRestoreArray(da,x,&xx);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,u,&uu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Evaluates \integral_{-1}^{1} f*v_i  where v_i is the ith basis polynomial via the GLL nodes and weights, since the v_i
      basis function is zero at all nodes except the ith one the integral is simply the weight_i * f(node_i)
*/
PetscErrorCode ComputeRhs(DM da,PetscGLL *gll,Vec b)
{
  PetscErrorCode ierr;
  PetscInt       i,j,xs,xn,n = gll->n;
  PetscScalar    *bb,*xx;
  PetscReal      xd;
  Vec            blocal,xlocal;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);
  xs   = xs/(n-1);
  xn   = xn/(n-1);
  ierr = DMGetLocalVector(da,&blocal);CHKERRQ(ierr);
  ierr = VecZeroEntries(blocal);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,blocal,&bb);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&xlocal);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,xlocal,&xx);CHKERRQ(ierr);
  /* loop over local spectral elements */
  for (j=xs; j<xs+xn; j++) {
    /* loop over GLL points in each element */
    for (i=0; i<n; i++) {
      xd              = xx[j*(n-1) + i];
      bb[j*(n-1) + i] += -gll->weights[i]*(-20.*PETSC_PI*xd*PetscSinReal(5.*PETSC_PI*xd) + (2. - (5.*PETSC_PI)*(5.*PETSC_PI)*(xd*xd - 1.))*PetscCosReal(5.*PETSC_PI*xd));
    }
  }
  ierr = DMDAVecRestoreArray(da,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,blocal,&bb);CHKERRQ(ierr);
  ierr = VecZeroEntries(b);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,blocal,ADD_VALUES,b);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,blocal,ADD_VALUES,b);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&blocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Run with -build_twosided allreduce -pc_type bjacobi -sub_pc_type lu -q 16 -ksp_rtol 1.e-34 (or 1.e-14 for double precision)

     -q <q> number of spectral elements to use
     -N <N> maximum number of GLL points per element

*/
int main(int argc,char **args)
{
  PetscErrorCode ierr;
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

  ierr = PetscInitialize(&argc,&args,NULL,NULL);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-q",&q,NULL);CHKERRQ(ierr);

  ierr = PetscDrawCreate(PETSC_COMM_WORLD,NULL,"Log(Error norm) vs Number of GLL points",0,0,500,500,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(draw,1,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGSetUseMarkers(lg,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,NULL,"Number of GLL points","Log(Error Norm)");CHKERRQ(ierr);

  for (n=4; n<N; n+=2) {

    /*
       da contains the information about the parallel layout of the elements
    */
    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,q*(n-1)+1,1,1,NULL,&da);CHKERRQ(ierr);
    ierr = DMSetFromOptions(da);CHKERRQ(ierr);
    ierr = DMSetUp(da);CHKERRQ(ierr);
    ierr = DMDAGetInfo(da,NULL,&q,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    q = (q-1)/(n-1);  /* number of spectral elements */

    /*
       gll simply contains the GLL node and weight values
    */
    ierr = PetscMalloc2(n,&gll.nodes,n,&gll.weights);CHKERRQ(ierr);
    ierr = PetscDTGaussLobattoLegendreQuadrature(n,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,gll.nodes,gll.weights);CHKERRQ(ierr);
    gll.n = n;
    ierr = DMDASetGLLCoordinates(da,gll.n,gll.nodes);CHKERRQ(ierr);

    /*
       Creates the element stiffness matrix for the given gll
    */
    ierr = PetscGaussLobattoLegendreElementLaplacianCreate(gll.n,gll.nodes,gll.weights,&A);CHKERRQ(ierr);

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
    ierr = DMCreateMatrix(da,&K);CHKERRQ(ierr);
    ierr = MatSetOption(K,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);
    xs   = xs/(n-1);
    xn   = xn/(n-1);
    ierr = PetscMalloc1(n,&rows);CHKERRQ(ierr);
    /*
        loop over local elements
    */
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<n; l++) rows[l] = j*(n-1)+l;
      ierr = MatSetValues(K,n,rows,n,rows,&A[0][0],ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatCreateVecs(K,&x,&b);CHKERRQ(ierr);
    ierr = ComputeRhs(da,&gll,b);CHKERRQ(ierr);

    /*
        Replace the first and last rows/columns of the matrix with the identity to obtain the zero Dirichlet boundary conditions
    */
    rows[0] = 0;
    rows[1] = q*(n-1);
    ierr = MatZeroRowsColumns(K,2,rows,1.0,x,b);CHKERRQ(ierr);
    ierr = PetscFree(rows);CHKERRQ(ierr);

    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,K,K);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

    /* compute the error to the continium problem */
    ierr = ComputeSolution(da,&gll,b);CHKERRQ(ierr);
    ierr = VecAXPY(x,-1.0,b);CHKERRQ(ierr);

    /* compute the L^2 norm of the error */
    ierr = VecGetArray(x,&f);CHKERRQ(ierr);
    ierr = PetscGaussLobattoLegendreIntegrate(gll.n,gll.nodes,gll.weights,f,&norm);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&f);CHKERRQ(ierr);
    norm = PetscSqrtReal(norm);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"L^2 norm of the error %D %g\n",n,(double)norm);CHKERRQ(ierr);
    PetscCheckFalse(n > 10 && norm > 1.e-8,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Slower convergence than expected");
    xc   = (PetscReal)n;
    yc   = PetscLog10Real(norm);
    ierr = PetscDrawLGAddPoint(lg,&xc,&yc);CHKERRQ(ierr);
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&K);CHKERRQ(ierr);
    ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(gll.n,gll.nodes,gll.weights,&A);CHKERRQ(ierr);
    ierr = PetscFree2(gll.nodes,gll.weights);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
  }
  ierr = PetscDrawLGDestroy(&lg);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
      requires: !complex

   test:
     requires: !single

   test:
     suffix: 2
     nsize: 2
     requires: !single superlu_dist

TEST*/
