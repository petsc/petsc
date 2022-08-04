
static char help[] = "Tests matrix ordering routines.\n\n";

#include <petscmat.h>
extern PetscErrorCode MatGetOrdering_myordering(Mat,MatOrderingType,IS*,IS*);

int main(int argc,char **args)
{
  Mat               C,Cperm;
  PetscInt          i,j,m = 5,n = 5,Ii,J,ncols;
  PetscScalar       v;
  PetscMPIInt       size;
  IS                rperm,cperm,icperm;
  const PetscInt    *rperm_ptr,*cperm_ptr,*cols;
  const PetscScalar *vals;
  PetscBool         TestMyorder=PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* create the matrix for the five point stencil, YET AGAIN */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&C));
  PetscCall(MatSetUp(C));
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetOrdering(C,MATORDERINGND,&rperm,&cperm));
  PetscCall(ISView(rperm,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(ISDestroy(&rperm));
  PetscCall(ISDestroy(&cperm));

  PetscCall(MatGetOrdering(C,MATORDERINGRCM,&rperm,&cperm));
  PetscCall(ISView(rperm,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(ISDestroy(&rperm));
  PetscCall(ISDestroy(&cperm));

  PetscCall(MatGetOrdering(C,MATORDERINGQMD,&rperm,&cperm));
  PetscCall(ISView(rperm,PETSC_VIEWER_STDOUT_SELF));
  PetscCall(ISDestroy(&rperm));
  PetscCall(ISDestroy(&cperm));

  /* create Cperm = rperm*C*icperm */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-testmyordering",&TestMyorder,NULL));
  if (TestMyorder) {
    PetscCall(MatGetOrdering_myordering(C,MATORDERINGQMD,&rperm,&cperm));
    printf("myordering's rperm:\n");
    PetscCall(ISView(rperm,PETSC_VIEWER_STDOUT_SELF));
    PetscCall(ISInvertPermutation(cperm,PETSC_DECIDE,&icperm));
    PetscCall(ISGetIndices(rperm,&rperm_ptr));
    PetscCall(ISGetIndices(icperm,&cperm_ptr));
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&Cperm));
    for (i=0; i<m*n; i++) {
      PetscCall(MatGetRow(C,rperm_ptr[i],&ncols,&cols,&vals));
      for (j=0; j<ncols; j++) {
        /* printf(" (%d %d %g)\n",i,cperm_ptr[cols[j]],vals[j]); */
        PetscCall(MatSetValues(Cperm,1,&i,1,&cperm_ptr[cols[j]],&vals[j],INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(Cperm,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Cperm,MAT_FINAL_ASSEMBLY));
    PetscCall(ISRestoreIndices(rperm,&rperm_ptr));
    PetscCall(ISRestoreIndices(icperm,&cperm_ptr));

    PetscCall(ISDestroy(&rperm));
    PetscCall(ISDestroy(&cperm));
    PetscCall(ISDestroy(&icperm));
    PetscCall(MatDestroy(&Cperm));
  }

  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

#include <petsc/private/matimpl.h>
/* This is modified from MatGetOrdering_Natural() */
PetscErrorCode MatGetOrdering_myordering(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  PetscInt       n,i,*ii;
  PetscBool      done;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));
  PetscCall(MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,NULL,NULL,&done));
  PetscCall(MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,NULL,NULL,NULL,&done));
  if (done) { /* matrix may be "compressed" in symbolic factorization, due to i-nodes or block storage */
    PetscCall(PetscMalloc1(n,&ii));
    for (i=0; i<n; i++) ii[i] = n-i-1; /* replace your index here */
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_COPY_VALUES,irow));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_OWN_POINTER,icol));
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"MatRestoreRowIJ fails!");
  PetscCall(ISSetPermutation(*irow));
  PetscCall(ISSetPermutation(*icol));
  PetscFunctionReturn(0);
}

/*TEST

   test:

TEST*/
