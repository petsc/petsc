
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* create the matrix for the five point stencil, YET AGAIN */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERINGND,&rperm,&cperm);CHKERRQ(ierr);
  ierr = ISView(rperm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISDestroy(&rperm);CHKERRQ(ierr);
  ierr = ISDestroy(&cperm);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERINGRCM,&rperm,&cperm);CHKERRQ(ierr);
  ierr = ISView(rperm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISDestroy(&rperm);CHKERRQ(ierr);
  ierr = ISDestroy(&cperm);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERINGQMD,&rperm,&cperm);CHKERRQ(ierr);
  ierr = ISView(rperm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISDestroy(&rperm);CHKERRQ(ierr);
  ierr = ISDestroy(&cperm);CHKERRQ(ierr);

  /* create Cperm = rperm*C*icperm */
  ierr = PetscOptionsGetBool(NULL,NULL,"-testmyordering",&TestMyorder,NULL);CHKERRQ(ierr);
  if (TestMyorder) {
    ierr = MatGetOrdering_myordering(C,MATORDERINGQMD,&rperm,&cperm);CHKERRQ(ierr);
    printf("myordering's rperm:\n");
    ierr = ISView(rperm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = ISInvertPermutation(cperm,PETSC_DECIDE,&icperm);CHKERRQ(ierr);
    ierr = ISGetIndices(rperm,&rperm_ptr);CHKERRQ(ierr);
    ierr = ISGetIndices(icperm,&cperm_ptr);CHKERRQ(ierr);
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&Cperm);CHKERRQ(ierr);
    for (i=0; i<m*n; i++) {
      ierr = MatGetRow(C,rperm_ptr[i],&ncols,&cols,&vals);CHKERRQ(ierr);
      for (j=0; j<ncols; j++) {
        /* printf(" (%d %d %g)\n",i,cperm_ptr[cols[j]],vals[j]); */
        ierr = MatSetValues(Cperm,1,&i,1,&cperm_ptr[cols[j]],&vals[j],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(Cperm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Cperm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = ISRestoreIndices(rperm,&rperm_ptr);CHKERRQ(ierr);
    ierr = ISRestoreIndices(icperm,&cperm_ptr);CHKERRQ(ierr);

    ierr = ISDestroy(&rperm);CHKERRQ(ierr);
    ierr = ISDestroy(&cperm);CHKERRQ(ierr);
    ierr = ISDestroy(&icperm);CHKERRQ(ierr);
    ierr = MatDestroy(&Cperm);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

#include <petsc/private/matimpl.h>
/* This is modified from MatGetOrdering_Natural() */
PetscErrorCode MatGetOrdering_myordering(Mat mat,MatOrderingType type,IS *irow,IS *icol)
{
  PetscErrorCode ierr;
  PetscInt       n,i,*ii;
  PetscBool      done;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MatGetRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,&n,NULL,NULL,&done);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat,0,PETSC_FALSE,PETSC_TRUE,NULL,NULL,NULL,&done);CHKERRQ(ierr);
  if (done) { /* matrix may be "compressed" in symbolic factorization, due to i-nodes or block storage */
    ierr = PetscMalloc1(n,&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) ii[i] = n-i-1; /* replace your index here */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_COPY_VALUES,irow);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_OWN_POINTER,icol);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"MatRestoreRowIJ fails!");
  ierr = ISSetPermutation(*irow);CHKERRQ(ierr);
  ierr = ISSetPermutation(*icol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:

TEST*/
