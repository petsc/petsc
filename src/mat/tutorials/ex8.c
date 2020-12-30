
static char help[] = "Shows how to add a new MatOperation to AIJ MatType\n\n";

#include <petscmat.h>
#include <petscblaslapack.h>

static PetscErrorCode MatScaleUserImpl_SeqAIJ(Mat inA,PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatScale(inA,alpha);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatScaleUserImpl(Mat,PetscScalar);

static PetscErrorCode MatScaleUserImpl_MPIAIJ(Mat A,PetscScalar aa)
{
  PetscErrorCode ierr;
  Mat            AA,AB;

  PetscFunctionBegin;
  ierr = MatMPIAIJGetSeqAIJ(A,&AA,&AB,NULL);CHKERRQ(ierr);
  ierr = MatScaleUserImpl(AA,aa);CHKERRQ(ierr);
  ierr = MatScaleUserImpl(AB,aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This routine registers MatScaleUserImpl_SeqAIJ() and
   MatScaleUserImpl_MPIAIJ() as methods providing MatScaleUserImpl()
   functionality for SeqAIJ and MPIAIJ matrix-types */
PetscErrorCode RegisterMatScaleUserImpl(Mat mat)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat), &size);CHKERRMPI(ierr);

  if (size == 1) { /* SeqAIJ Matrix */
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatScaleUserImpl_C",MatScaleUserImpl_SeqAIJ);CHKERRQ(ierr);

  } else { /* MPIAIJ Matrix */
    Mat AA,AB;
    ierr = MatMPIAIJGetSeqAIJ(mat,&AA,&AB,NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)mat,"MatScaleUserImpl_C",MatScaleUserImpl_MPIAIJ);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)AA,"MatScaleUserImpl_C",MatScaleUserImpl_SeqAIJ);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)AB,"MatScaleUserImpl_C",MatScaleUserImpl_SeqAIJ);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* this routines queries the already registered MatScaleUserImp_XXX
   implementations for the given matrix, and calls the correct
   routine. i.e if MatType is SeqAIJ, MatScaleUserImpl_SeqAIJ() gets
   called, and if MatType is MPIAIJ, MatScaleUserImpl_MPIAIJ() gets
   called */
PetscErrorCode MatScaleUserImpl(Mat mat,PetscScalar a)
{
  PetscErrorCode ierr,(*f)(Mat,PetscScalar);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatScaleUserImpl_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Main user code that uses MatScaleUserImpl() */

int main(int argc,char **args)
{
  Mat            mat;
  PetscInt       i,j,m = 2,n,Ii,J;
  PetscErrorCode ierr;
  PetscScalar    v,none = -1.0;
  PetscMPIInt    rank,size;


  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  n    = 2*size;

  /* create the matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);

  /* register user defined MatScaleUser() operation for both SeqAIJ
     and MPIAIJ types */
  ierr = RegisterMatScaleUserImpl(mat);CHKERRQ(ierr);

  /* assemble the matrix */
  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(mat,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(mat,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(mat,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(mat,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(mat,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* check the matrix before and after scaling by -1.0 */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix _before_ MatScaleUserImpl() operation\n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatScaleUserImpl(mat,none);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix _after_ MatScaleUserImpl() operation\n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

   test:
      suffix: 2
      nsize: 2

TEST*/
