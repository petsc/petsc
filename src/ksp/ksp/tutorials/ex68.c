
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
  PetscInt       i,m;
  PetscScalar    *xx;
  PetscReal      xd;

  PetscFunctionBegin;
  PetscCall(VecGetSize(x,&m));
  PetscCall(VecGetArray(x,&xx));
  for (i=0; i<m; i++) {
    xd    = nodes[i];
    xx[i] = (xd*xd - 1.0)*PetscCosReal(5.*PETSC_PI*xd);
  }
  PetscCall(VecRestoreArray(x,&xx));
  PetscFunctionReturn(0);
}

/*
      Evaluates \integral_{-1}^{1} f*v_i  where v_i is the ith basis polynomial via the GLL nodes and weights, since the v_i
      basis function is zero at all nodes except the ith one the integral is simply the weight_i * f(node_i)
*/
PetscErrorCode ComputeRhs(PetscInt n,PetscReal *nodes,PetscReal *weights,Vec b)
{
  PetscInt       i,m;
  PetscScalar    *bb;
  PetscReal      xd;

  PetscFunctionBegin;
  PetscCall(VecGetSize(b,&m));
  PetscCall(VecGetArray(b,&bb));
  for (i=0; i<m; i++) {
    xd    = nodes[i];
    bb[i] = -weights[i]*(-20.*PETSC_PI*xd*PetscSinReal(5.*PETSC_PI*xd) + (2. - (5.*PETSC_PI)*(5.*PETSC_PI)*(xd*xd - 1.))*PetscCosReal(5.*PETSC_PI*xd));
  }
  PetscCall(VecRestoreArray(b,&bb));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
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

  PetscCall(PetscInitialize(&argc,&args,NULL,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));

  PetscCall(PetscDrawCreate(PETSC_COMM_SELF,NULL,"Log(Error norm) vs Number of GLL points",0,0,500,500,&draw));
  PetscCall(PetscDrawSetFromOptions(draw));
  PetscCall(PetscDrawLGCreate(draw,1,&lg));
  PetscCall(PetscDrawLGSetUseMarkers(lg,PETSC_TRUE));
  PetscCall(PetscDrawLGGetAxis(lg,&axis));
  PetscCall(PetscDrawAxisSetLabels(axis,NULL,"Number of GLL points","Log(Error Norm)"));

  for (n=4; n<N; n+=2) {
    /*
       compute GLL node and weight values
    */
    PetscCall(PetscMalloc2(n,&nodes,n,&weights));
    PetscCall(PetscDTGaussLobattoLegendreQuadrature(n,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,nodes,weights));
    /*
       Creates the element stiffness matrix for the given gll
    */
    PetscCall(PetscGaussLobattoLegendreElementLaplacianCreate(n,nodes,weights,&A));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n,n,&A[0][0],&K));
    rows[0] = 0;
    rows[1] = n-1;
    PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
    PetscCall(MatCreateVecs(K,&x,&b));
    PetscCall(ComputeRhs(n,nodes,weights,b));
    /*
        Replace the first and last rows/columns of the matrix with the identity to obtain the zero Dirichlet boundary conditions
    */
    PetscCall(MatZeroRowsColumns(K,2,rows,1.0,x,b));
    PetscCall(KSPSetOperators(ksp,K,K));
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCLU));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp,b,x));

    /* compute the error to the continium problem */
    PetscCall(ComputeSolution(n,nodes,weights,b));
    PetscCall(VecAXPY(x,-1.0,b));

    /* compute the L^2 norm of the error */
    PetscCall(VecGetArray(x,&f));
    PetscCall(PetscGaussLobattoLegendreIntegrate(n,nodes,weights,f,&norm));
    PetscCall(VecRestoreArray(x,&f));
    norm = PetscSqrtReal(norm);
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"L^2 norm of the error %D %g\n",n,(double)norm));
    xc   = (PetscReal)n;
    yc   = PetscLog10Real(norm);
    PetscCall(PetscDrawLGAddPoint(lg,&xc,&yc));
    PetscCall(PetscDrawLGDraw(lg));

    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&K));
    PetscCall(PetscGaussLobattoLegendreElementLaplacianDestroy(n,nodes,weights,&A));
    PetscCall(PetscFree2(nodes,weights));
  }
  PetscCall(PetscDrawSetPause(draw,-2));
  PetscCall(PetscDrawLGDestroy(&lg));
  PetscCall(PetscDrawDestroy(&draw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
      requires: !complex

   test:

TEST*/
