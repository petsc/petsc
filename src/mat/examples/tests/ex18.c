static char help[] = "Tests the use of MatZeroRowsColumns() for parallel matrices.\n\
Contributed-by: Stephan Kramer <s.kramer@imperial.ac.uk>\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  Vec            x, rhs, y;
  PetscInt       i,j,k,b,m = 3,n,nlocal=2,bs=1,Ii,J;
  PetscInt       *boundary_nodes, nboundary_nodes, *boundary_indices;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscScalar    v,v0,v1,v2,a0=0.1,a,rhsval, *boundary_values,diag = 1.0;
  PetscReal      norm;
  char           convname[64];
  PetscBool      upwind = PETSC_FALSE, nonlocalBC = PETSC_FALSE, zerorhs = PETSC_TRUE, convert = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n = nlocal*size;

  ierr = PetscOptionsGetInt(NULL,NULL, "-bs", &bs, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL, "-nonlocal_bc", &nonlocalBC, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL, "-diag", &diag, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-convname",convname,sizeof(convname),&convert);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL, "-zerorhs", &zerorhs, NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n*bs,m*n*bs);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &rhs);CHKERRQ(ierr);
  ierr = VecSetSizes(rhs, PETSC_DECIDE, m*n*bs);CHKERRQ(ierr);
  ierr = VecSetFromOptions(rhs);CHKERRQ(ierr);
  ierr = VecSetUp(rhs);CHKERRQ(ierr);

  rhsval = 0.0;
  for (i=0; i<m; i++) {
    for (j=nlocal*rank; j<nlocal*(rank+1); j++) {
      a = a0;
      for (b=0; b<bs; b++) {
        /* let's start with a 5-point stencil diffusion term */
        v = -1.0;  Ii = (j + n*i)*bs + b;
        if (i>0)   {J = Ii - n*bs; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (i<m-1) {J = Ii + n*bs; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j>0)   {J = Ii - 1*bs; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        if (j<n-1) {J = Ii + 1*bs; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
        if (upwind) {
          /* now add a 2nd order upwind advection term to add a little asymmetry */
          if (j>2) {
            J = Ii-2*bs; v2 = 0.5*a; v1 = -2.0*a; v0 = 1.5*a;
            ierr = MatSetValues(A,1,&Ii,1,&J,&v2,ADD_VALUES);CHKERRQ(ierr);
          } else {
            /* fall back to 1st order upwind */
            v1 = -1.0*a; v0 = 1.0*a;
          };
          if (j>1) {J = Ii-1*bs; ierr = MatSetValues(A,1,&Ii,1,&J,&v1,ADD_VALUES);CHKERRQ(ierr);}
          ierr = MatSetValues(A,1,&Ii,1,&Ii,&v0,ADD_VALUES);CHKERRQ(ierr);
          a /= 10.; /* use a different velocity for the next component */
          /* add a coupling to the previous and next components */
          v = 0.5;
          if (b>0) {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
          if (b<bs-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
        }
        /* make up some rhs */
        ierr = VecSetValue(rhs, Ii, rhsval, INSERT_VALUES);CHKERRQ(ierr);
        rhsval += 1.0;
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (convert) { /* Test different Mat implementations */
    Mat B;

    ierr = MatConvert(A,convname,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    A    = B;
  }

  ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);
  /* set rhs to zero to simplify */
  if (zerorhs) {
    ierr = VecZeroEntries(rhs);CHKERRQ(ierr);
  }

  if (nonlocalBC) {
    /*version where boundary conditions are set by processes that don't necessarily own the nodes */
    if (!rank) {
      nboundary_nodes = size>m ? nlocal : m-size+nlocal;
      ierr = PetscMalloc1(nboundary_nodes,&boundary_nodes);CHKERRQ(ierr);
      k = 0;
      for (i=size; i<m; i++,k++) {boundary_nodes[k] = n*i;};
    } else if (rank < m) {
      nboundary_nodes = nlocal+1;
      ierr = PetscMalloc1(nboundary_nodes,&boundary_nodes);CHKERRQ(ierr);
      boundary_nodes[0] = rank*n;
      k = 1;
    } else {
      nboundary_nodes = nlocal;
      ierr = PetscMalloc1(nboundary_nodes,&boundary_nodes);CHKERRQ(ierr);
      k = 0;
    }
    for (j=nlocal*rank; j<nlocal*(rank+1); j++,k++) {boundary_nodes[k] = j;};
  } else {
    /*version where boundary conditions are set by the node owners only */
    ierr = PetscMalloc1(m*n,&boundary_nodes);CHKERRQ(ierr);
    k=0;
    for (j=0; j<n; j++) {
      Ii = j;
      if (Ii>=rank*m*nlocal && Ii<(rank+1)*m*nlocal) boundary_nodes[k++] = Ii;
    }
    for (i=1; i<m; i++) {
      Ii = n*i;
      if (Ii>=rank*m*nlocal && Ii<(rank+1)*m*nlocal) boundary_nodes[k++] = Ii;
    }
    nboundary_nodes = k;
  }

  ierr = VecDuplicate(rhs, &x);CHKERRQ(ierr);
  ierr = VecZeroEntries(x);CHKERRQ(ierr);
  ierr = PetscMalloc2(nboundary_nodes*bs,&boundary_indices,nboundary_nodes*bs,&boundary_values);CHKERRQ(ierr);
  for (k=0; k<nboundary_nodes; k++) {
    Ii = boundary_nodes[k]*bs;
    v = 1.0*boundary_nodes[k];
    for (b=0; b<bs; b++, Ii++) {
      boundary_indices[k*bs+b] = Ii;
      boundary_values[k*bs+b] = v;
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d %D %f\n", rank, Ii, (double)PetscRealPart(v));CHKERRQ(ierr);
      v += 0.1;
    }
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
  ierr = VecSetValues(x, nboundary_nodes*bs, boundary_indices, boundary_values, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* We can check the rhs returned by MatZeroColumns by computing y=rhs-A*x  and overwriting the boundary entries with boundary values */
  ierr = VecDuplicate(x, &y);CHKERRQ(ierr);
  ierr = MatMult(A, x, y);CHKERRQ(ierr);
  ierr = VecAYPX(y, -1.0, rhs);CHKERRQ(ierr);
  for (k=0; k<nboundary_nodes*bs; k++) boundary_values[k] *= diag;
  ierr = VecSetValues(y, nboundary_nodes*bs, boundary_indices, boundary_values, INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "*** Matrix A and vector x:\n");CHKERRQ(ierr);
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatZeroRowsColumns(A, nboundary_nodes*bs, boundary_indices, diag, x, rhs);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "*** Vector rhs returned by MatZeroRowsColumns\n");CHKERRQ(ierr);
  ierr = VecView(rhs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecAXPY(y, -1.0, rhs);CHKERRQ(ierr);
  ierr = VecNorm(y, NORM_INFINITY, &norm);CHKERRQ(ierr);
  if (norm > 1.0e-10) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "*** Difference between rhs and y, inf-norm: %f\n", (double)norm);CHKERRQ(ierr);
    ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Bug in MatZeroRowsColumns");
  }

  ierr = PetscFree(boundary_nodes);CHKERRQ(ierr);
  ierr = PetscFree2(boundary_indices,boundary_values);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      suffix: 0

   test:
      suffix: 1
      nsize: 2

   test:
      suffix: 10
      nsize: 2
      args: -bs 2 -nonlocal_bc

   test:
      suffix: 11
      nsize: 7
      args: -bs 2 -nonlocal_bc

   test:
      suffix: 12
      args: -bs 2 -nonlocal_bc -mat_type baij

   test:
      suffix: 13
      nsize: 2
      args: -bs 2 -nonlocal_bc -mat_type baij

   test:
      suffix: 14
      nsize: 7
      args: -bs 2 -nonlocal_bc -mat_type baij

   test:
      suffix: 2
      nsize: 7

   test:
      suffix: 3
      args: -mat_type baij

   test:
      suffix: 4
      nsize: 2
      args: -mat_type baij

   test:
      suffix: 5
      nsize: 7
      args: -mat_type baij

   test:
      suffix: 6
      args: -bs 2 -mat_type baij

   test:
      suffix: 7
      nsize: 2
      args: -bs 2 -mat_type baij

   test:
      suffix: 8
      nsize: 7
      args: -bs 2 -mat_type baij

   test:
      suffix: 9
      args: -bs 2 -nonlocal_bc

   test:
      suffix: 15
      args: -bs 2 -nonlocal_bc -convname shell

   test:
      suffix: 16
      nsize: 2
      args: -bs 2 -nonlocal_bc -convname shell

   test:
      suffix: 17
      args: -bs 2 -nonlocal_bc -convname dense

   testset:
      suffix: full
      nsize: {{1 3}separate output}
      args: -diag {{0.12 -0.13}separate output} -convname {{aij shell baij}separate output} -zerorhs 0
TEST*/
