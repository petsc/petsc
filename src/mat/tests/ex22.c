
static char help[] = "Tests matrix ordering routines.\n\n";

#include <petscmat.h>
extern PetscErrorCode MatGetOrdering_myordering(Mat,MatOrderingType,IS*,IS*);

int main(int argc,char **args)
{
  Mat               C,Cperm;
  PetscInt          i,j,m = 5,n = 5,Ii,J,ncols;
  PetscErrorCode    ierr;
  PetscScalar       v;
  PetscMPIInt       size;
  IS                rperm,cperm,icperm;
  const PetscInt    *rperm_ptr,*cperm_ptr,*cols;
  const PetscScalar *vals;
  PetscBool         TestMyorder=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* create the matrix for the five point stencil, YET AGAIN */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&C));
  CHKERRQ(MatSetUp(C));
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatGetOrdering(C,MATORDERINGND,&rperm,&cperm));
  CHKERRQ(ISView(rperm,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(ISDestroy(&rperm));
  CHKERRQ(ISDestroy(&cperm));

  CHKERRQ(MatGetOrdering(C,MATORDERINGRCM,&rperm,&cperm));
  CHKERRQ(ISView(rperm,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(ISDestroy(&rperm));
  CHKERRQ(ISDestroy(&cperm));

  CHKERRQ(MatGetOrdering(C,MATORDERINGQMD,&rperm,&cperm));
  CHKERRQ(ISView(rperm,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(ISDestroy(&rperm));
  CHKERRQ(ISDestroy(&cperm));

  /* create Cperm = rperm*C*icperm */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-testmyordering",&TestMyorder,NULL));
  if (TestMyorder) {
    CHKERRQ(MatGetOrdering_myordering(C,MATORDERINGQMD,&rperm,&cperm));
    printf("myordering's rperm:\n");
    CHKERRQ(ISView(rperm,PETSC_VIEWER_STDOUT_SELF));
    CHKERRQ(ISInvertPermutation(cperm,PETSC_DECIDE,&icperm));
    CHKERRQ(ISGetIndices(rperm,&rperm_ptr));
    CHKERRQ(ISGetIndices(icperm,&cperm_ptr));
    CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&Cperm));
    for (i=0; i<m*n; i++) {
      CHKERRQ(MatGetRow(C,rperm_ptr[i],&ncols,&cols,&vals));
      for (j=0; j<ncols; j++) {
        /* printf(" (%d %d %g)\n",i,cperm_ptr[cols[j]],vals[j]); */
        CHKERRQ(MatSetValues(Cperm,1,&i,1,&cperm_ptr[cols[j]],&vals[j],INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(Cperm,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Cperm,MAT_FINAL_ASSEMBLY));
    CHKERRQ(ISRestoreIndices(rperm,&rperm_ptr));
    CHKERRQ(ISRestoreIndices(icperm,&cperm_ptr));

    CHKERRQ(ISDestroy(&rperm));
    CHKERRQ(ISDestroy(&cperm));
    CHKERRQ(ISDestroy(&icperm));
    CHKERRQ(MatDestroy(&Cperm));
  }

  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

#include <petsc/private/matimpl.h>
/* This is modified from MatGetOrdering_Natural() */
PetscErrorCode MatGetOrdering_myordering(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  PetscInt       n,i,*ii;
  PetscBool      done;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)mat,&comm));
  CHKERRQ(MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,NULL,NULL,&done));
  CHKERRQ(MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,NULL,NULL,NULL,&done));
  if (done) { /* matrix may be "compressed" in symbolic factorization, due to i-nodes or block storage */
    CHKERRQ(PetscMalloc1(n,&ii));
    for (i=0; i<n; i++) ii[i] = n-i-1; /* replace your index here */
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_COPY_VALUES,irow));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_OWN_POINTER,icol));
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"MatRestoreRowIJ fails!");
  CHKERRQ(ISSetPermutation(*irow));
  CHKERRQ(ISSetPermutation(*icol));
  PetscFunctionReturn(0);
}

/*TEST

   test:

TEST*/
