
#include <petscdt.h>
#include <petscdraw.h>
#include <petscviewer.h>
#include <petscksp.h>

/*
      Solves -Laplacian u = f,  u(-1) = u(1) = 0 with a single spectral element for n = 4 to N GLL points

      Plots the L_2 norm of the error (evaluated via the GLL nodes and weights) as a function of n.

*/
PetscErrorCode ComputeSolution(PetscInt n,PetscReal *nodes,PetscReal *weights,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       i,m;
  PetscScalar    *xx;
  PetscReal      xd;

  PetscFunctionBegin;
  CHKERRQ(VecGetSize(x,&m));
  CHKERRQ(VecGetArray(x,&xx));
  for (i=0; i<m; i++) {
    xd    = nodes[i];
    xx[i] = (xd*xd - 1.0)*PetscCosReal(5.*PETSC_PI*xd);
  }
  CHKERRQ(VecRestoreArray(x,&xx));
  PetscFunctionReturn(0);
}

/*
      Evaluates \integral_{-1}^{1} f*v_i  where v_i is the ith basis polynomial via the GLL nodes and weights, since the v_i
      basis function is zero at all nodes except the ith one the integral is simply the weight_i * f(node_i)
*/
PetscErrorCode ComputeRhs(PetscInt n,PetscReal *nodes,PetscReal *weights,Vec b)
{
  PetscErrorCode ierr;
  PetscInt       i,m;
  PetscScalar    *bb;
  PetscReal      xd;

  PetscFunctionBegin;
  CHKERRQ(VecGetSize(b,&m));
  CHKERRQ(VecGetArray(b,&bb));
  for (i=0; i<m; i++) {
    xd    = nodes[i];
    bb[i] = -weights[i]*(-20.*PETSC_PI*xd*PetscSinReal(5.*PETSC_PI*xd) + (2. - (5.*PETSC_PI)*(5.*PETSC_PI)*(xd*xd - 1.))*PetscCosReal(5.*PETSC_PI*xd));
  }
  CHKERRQ(VecRestoreArray(b,&bb));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscReal      *nodes;
  PetscReal      *weights;
  PetscInt       N = 80,n;
  PetscReal      **A;
  Mat            K;
  KSP            ksp;
  PC             pc;
  Vec            x,b;
  PetscInt       rows[2];
  PetscReal      norm,xc,yc;
  PetscScalar    *f;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscDrawAxis  axis;

  CHKERRQ(PetscInitialize(&argc,&args,NULL,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));

  CHKERRQ(PetscDrawCreate(PETSC_COMM_SELF,NULL,"Log(Error norm) vs Number of GLL points",0,0,500,500,&draw));
  CHKERRQ(PetscDrawSetFromOptions(draw));
  CHKERRQ(PetscDrawLGCreate(draw,1,&lg));
  CHKERRQ(PetscDrawLGSetUseMarkers(lg,PETSC_TRUE));
  CHKERRQ(PetscDrawLGGetAxis(lg,&axis));
  CHKERRQ(PetscDrawAxisSetLabels(axis,NULL,"Number of GLL points","Log(Error Norm)"));

  for (n=4; n<N; n+=2) {
    /*
       compute GLL node and weight values
    */
    CHKERRQ(PetscMalloc2(n,&nodes,n,&weights));
    CHKERRQ(PetscDTGaussLobattoLegendreQuadrature(n,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,nodes,weights));
    /*
       Creates the element stiffness matrix for the given gll
    */
    CHKERRQ(PetscGaussLobattoLegendreElementLaplacianCreate(n,nodes,weights,&A));
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,&A[0][0],&K));
    rows[0] = 0;
    rows[1] = n-1;
    CHKERRQ(KSPCreate(PETSC_COMM_SELF,&ksp));
    CHKERRQ(MatCreateVecs(K,&x,&b));
    CHKERRQ(ComputeRhs(n,nodes,weights,b));
    /*
        Replace the first and last rows/columns of the matrix with the identity to obtain the zero Dirichlet boundary conditions
    */
    CHKERRQ(MatZeroRowsColumns(K,2,rows,1.0,x,b));
    CHKERRQ(KSPSetOperators(ksp,K,K));
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCLU));
    CHKERRQ(KSPSetFromOptions(ksp));
    CHKERRQ(KSPSolve(ksp,b,x));

    /* compute the error to the continium problem */
    CHKERRQ(ComputeSolution(n,nodes,weights,b));
    CHKERRQ(VecAXPY(x,-1.0,b));

    /* compute the L^2 norm of the error */
    CHKERRQ(VecGetArray(x,&f));
    CHKERRQ(PetscGaussLobattoLegendreIntegrate(n,nodes,weights,f,&norm));
    CHKERRQ(VecRestoreArray(x,&f));
    norm = PetscSqrtReal(norm);
    CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"L^2 norm of the error %D %g\n",n,(double)norm));
    xc   = (PetscReal)n;
    yc   = PetscLog10Real(norm);
    CHKERRQ(PetscDrawLGAddPoint(lg,&xc,&yc));
    CHKERRQ(PetscDrawLGDraw(lg));

    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(KSPDestroy(&ksp));
    CHKERRQ(MatDestroy(&K));
    CHKERRQ(PetscGaussLobattoLegendreElementLaplacianDestroy(n,nodes,weights,&A));
    CHKERRQ(PetscFree2(nodes,weights));
  }
  CHKERRQ(PetscDrawSetPause(draw,-2));
  CHKERRQ(PetscDrawLGDestroy(&lg));
  CHKERRQ(PetscDrawDestroy(&draw));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  build:
      requires: !complex

   test:

TEST*/
