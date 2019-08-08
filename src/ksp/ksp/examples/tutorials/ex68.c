
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
  ierr = VecGetSize(x,&m);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    xd    = nodes[i];
    xx[i] = (xd*xd - 1.0)*PetscCosReal(5.*PETSC_PI*xd);
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
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
  ierr = VecGetSize(b,&m);CHKERRQ(ierr);
  ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    xd    = nodes[i];
    bb[i] = -weights[i]*(-20.*PETSC_PI*xd*PetscSinReal(5.*PETSC_PI*xd) + (2. - (5.*PETSC_PI)*(5.*PETSC_PI)*(xd*xd - 1.))*PetscCosReal(5.*PETSC_PI*xd));
  }
  ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
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

  ierr = PetscInitialize(&argc,&args,NULL,NULL);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);

  ierr = PetscDrawCreate(PETSC_COMM_SELF,NULL,"Log(Error norm) vs Number of GLL points",0,0,500,500,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(draw,1,&lg);CHKERRQ(ierr);
  ierr = PetscDrawLGSetUseMarkers(lg,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,NULL,"Number of GLL points","Log(Error Norm)");CHKERRQ(ierr);

  for (n=4; n<N; n+=2) {
    /*
       compute GLL node and weight values
    */
    ierr = PetscMalloc2(n,&nodes,n,&weights);CHKERRQ(ierr);
    ierr = PetscDTGaussLobattoLegendreQuadrature(n,PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,nodes,weights);
    /*
       Creates the element stiffness matrix for the given gll
    */
    ierr = PetscGaussLobattoLegendreElementLaplacianCreate(n,nodes,weights,&A);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,&A[0][0],&K);CHKERRQ(ierr);
    rows[0] = 0;
    rows[1] = n-1;
    ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
    ierr = MatCreateVecs(K,&x,&b);CHKERRQ(ierr);
    ierr = ComputeRhs(n,nodes,weights,b);CHKERRQ(ierr);
    /*
        Replace the first and last rows/columns of the matrix with the identity to obtain the zero Dirichlet boundary conditions
    */
    ierr = MatZeroRowsColumns(K,2,rows,1.0,x,b);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,K,K);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

    /* compute the error to the continium problem */
    ierr = ComputeSolution(n,nodes,weights,b);CHKERRQ(ierr);
    ierr = VecAXPY(x,-1.0,b);CHKERRQ(ierr);

    /* compute the L^2 norm of the error */
    ierr = VecGetArray(x,&f);CHKERRQ(ierr);
    ierr = PetscGaussLobattoLegendreIntegrate(n,nodes,weights,f,&norm);CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&f);CHKERRQ(ierr);
    norm = PetscSqrtReal(norm);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"L^2 norm of the error %D %g\n",n,(double)norm);CHKERRQ(ierr);
    xc   = (PetscReal)n;
    yc   = PetscLog10Real(norm);
    ierr = PetscDrawLGAddPoint(lg,&xc,&yc);CHKERRQ(ierr);
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&K);CHKERRQ(ierr);
    ierr = PetscGaussLobattoLegendreElementLaplacianDestroy(n,nodes,weights,&A);CHKERRQ(ierr);
    ierr = PetscFree2(nodes,weights);CHKERRQ(ierr);
  }
  ierr = PetscDrawSetPause(draw,-2);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(&lg);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
      requires: !complex

   test:

TEST*/
