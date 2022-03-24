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
  PetscScalar    v,v0,v1,v2,a0=0.1,a,rhsval, *boundary_values,diag = 1.0;
  PetscReal      norm;
  char           convname[64];
  PetscBool      upwind = PETSC_FALSE, nonlocalBC = PETSC_FALSE, zerorhs = PETSC_TRUE, convert = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n = nlocal*size;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-bs", &bs, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-nonlocal_bc", &nonlocalBC, NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL, "-diag", &diag, NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-convname",convname,sizeof(convname),&convert));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-zerorhs", &zerorhs, NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n*bs,m*n*bs));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A, NULL, &rhs));
  CHKERRQ(VecSetFromOptions(rhs));
  CHKERRQ(VecSetUp(rhs));

  rhsval = 0.0;
  for (i=0; i<m; i++) {
    for (j=nlocal*rank; j<nlocal*(rank+1); j++) {
      a = a0;
      for (b=0; b<bs; b++) {
        /* let's start with a 5-point stencil diffusion term */
        v = -1.0;  Ii = (j + n*i)*bs + b;
        if (i>0)   {J = Ii - n*bs; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
        if (i<m-1) {J = Ii + n*bs; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
        if (j>0)   {J = Ii - 1*bs; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
        if (j<n-1) {J = Ii + 1*bs; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
        v = 4.0; CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES));
        if (upwind) {
          /* now add a 2nd order upwind advection term to add a little asymmetry */
          if (j>2) {
            J = Ii-2*bs; v2 = 0.5*a; v1 = -2.0*a; v0 = 1.5*a;
            CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v2,ADD_VALUES));
          } else {
            /* fall back to 1st order upwind */
            v1 = -1.0*a; v0 = 1.0*a;
          };
          if (j>1) {J = Ii-1*bs; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v1,ADD_VALUES));}
          CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v0,ADD_VALUES));
          a /= 10.; /* use a different velocity for the next component */
          /* add a coupling to the previous and next components */
          v = 0.5;
          if (b>0) {J = Ii - 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
          if (b<bs-1) {J = Ii + 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
        }
        /* make up some rhs */
        CHKERRQ(VecSetValue(rhs, Ii, rhsval, INSERT_VALUES));
        rhsval += 1.0;
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  if (convert) { /* Test different Mat implementations */
    Mat B;

    CHKERRQ(MatConvert(A,convname,MAT_INITIAL_MATRIX,&B));
    CHKERRQ(MatDestroy(&A));
    A    = B;
  }

  CHKERRQ(VecAssemblyBegin(rhs));
  CHKERRQ(VecAssemblyEnd(rhs));
  /* set rhs to zero to simplify */
  if (zerorhs) {
    CHKERRQ(VecZeroEntries(rhs));
  }

  if (nonlocalBC) {
    /*version where boundary conditions are set by processes that don't necessarily own the nodes */
    if (rank == 0) {
      nboundary_nodes = size>m ? nlocal : m-size+nlocal;
      CHKERRQ(PetscMalloc1(nboundary_nodes,&boundary_nodes));
      k = 0;
      for (i=size; i<m; i++,k++) {boundary_nodes[k] = n*i;};
    } else if (rank < m) {
      nboundary_nodes = nlocal+1;
      CHKERRQ(PetscMalloc1(nboundary_nodes,&boundary_nodes));
      boundary_nodes[0] = rank*n;
      k = 1;
    } else {
      nboundary_nodes = nlocal;
      CHKERRQ(PetscMalloc1(nboundary_nodes,&boundary_nodes));
      k = 0;
    }
    for (j=nlocal*rank; j<nlocal*(rank+1); j++,k++) {boundary_nodes[k] = j;};
  } else {
    /*version where boundary conditions are set by the node owners only */
    CHKERRQ(PetscMalloc1(m*n,&boundary_nodes));
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

  CHKERRQ(VecDuplicate(rhs, &x));
  CHKERRQ(VecZeroEntries(x));
  CHKERRQ(PetscMalloc2(nboundary_nodes*bs,&boundary_indices,nboundary_nodes*bs,&boundary_values));
  for (k=0; k<nboundary_nodes; k++) {
    Ii = boundary_nodes[k]*bs;
    v = 1.0*boundary_nodes[k];
    for (b=0; b<bs; b++, Ii++) {
      boundary_indices[k*bs+b] = Ii;
      boundary_values[k*bs+b] = v;
      CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%d %" PetscInt_FMT " %f\n", rank, Ii, (double)PetscRealPart(v)));
      v += 0.1;
    }
  }
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL));
  CHKERRQ(VecSetValues(x, nboundary_nodes*bs, boundary_indices, boundary_values, INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /* We can check the rhs returned by MatZeroColumns by computing y=rhs-A*x  and overwriting the boundary entries with boundary values */
  CHKERRQ(VecDuplicate(x, &y));
  CHKERRQ(MatMult(A, x, y));
  CHKERRQ(VecAYPX(y, -1.0, rhs));
  for (k=0; k<nboundary_nodes*bs; k++) boundary_values[k] *= diag;
  CHKERRQ(VecSetValues(y, nboundary_nodes*bs, boundary_indices, boundary_values, INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "*** Matrix A and vector x:\n"));
  CHKERRQ(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatZeroRowsColumns(A, nboundary_nodes*bs, boundary_indices, diag, x, rhs));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "*** Vector rhs returned by MatZeroRowsColumns\n"));
  CHKERRQ(VecView(rhs,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecAXPY(y, -1.0, rhs));
  CHKERRQ(VecNorm(y, NORM_INFINITY, &norm));
  if (norm > 1.0e-10) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "*** Difference between rhs and y, inf-norm: %f\n", (double)norm));
    CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Bug in MatZeroRowsColumns");
  }

  CHKERRQ(PetscFree(boundary_nodes));
  CHKERRQ(PetscFree2(boundary_indices,boundary_values));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&rhs));
  CHKERRQ(MatDestroy(&A));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      diff_args: -j
      suffix: 0

   test:
      diff_args: -j
      suffix: 1
      nsize: 2

   test:
      diff_args: -j
      suffix: 10
      nsize: 2
      args: -bs 2 -nonlocal_bc

   test:
      diff_args: -j
      suffix: 11
      nsize: 7
      args: -bs 2 -nonlocal_bc

   test:
      diff_args: -j
      suffix: 12
      args: -bs 2 -nonlocal_bc -mat_type baij

   test:
      diff_args: -j
      suffix: 13
      nsize: 2
      args: -bs 2 -nonlocal_bc -mat_type baij

   test:
      diff_args: -j
      suffix: 14
      nsize: 7
      args: -bs 2 -nonlocal_bc -mat_type baij

   test:
      diff_args: -j
      suffix: 2
      nsize: 7

   test:
      diff_args: -j
      suffix: 3
      args: -mat_type baij

   test:
      diff_args: -j
      suffix: 4
      nsize: 2
      args: -mat_type baij

   test:
      diff_args: -j
      suffix: 5
      nsize: 7
      args: -mat_type baij

   test:
      diff_args: -j
      suffix: 6
      args: -bs 2 -mat_type baij

   test:
      diff_args: -j
      suffix: 7
      nsize: 2
      args: -bs 2 -mat_type baij

   test:
      diff_args: -j
      suffix: 8
      nsize: 7
      args: -bs 2 -mat_type baij

   test:
      diff_args: -j
      suffix: 9
      args: -bs 2 -nonlocal_bc

   test:
      diff_args: -j
      suffix: 15
      args: -bs 2 -nonlocal_bc -convname shell

   test:
      diff_args: -j
      suffix: 16
      nsize: 2
      args: -bs 2 -nonlocal_bc -convname shell

   test:
      diff_args: -j
      suffix: 17
      args: -bs 2 -nonlocal_bc -convname dense

   testset:
      diff_args: -j
      suffix: full
      nsize: {{1 3}separate output}
      args: -diag {{0.12 -0.13}separate output} -convname {{aij shell baij}separate output} -zerorhs 0

   test:
      diff_args: -j
      requires: cuda
      suffix: cusparse_1
      nsize: 1
      args: -diag {{0.12 -0.13}separate output} -convname {{seqaijcusparse mpiaijcusparse}separate output} -zerorhs 0 -mat_type {{seqaijcusparse mpiaijcusparse}separate output}

   test:
      diff_args: -j
      requires: cuda
      suffix: cusparse_3
      nsize: 3
      args: -diag {{0.12 -0.13}separate output} -convname mpiaijcusparse -zerorhs 0 -mat_type mpiaijcusparse

TEST*/
