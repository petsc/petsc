
static char help[] = "Shows how to add a new MatOperation to AIJ MatType\n\n";

#include <petscmat.h>
#include <petscblaslapack.h>

static PetscErrorCode MatScaleUserImpl_SeqAIJ(Mat inA,PetscScalar alpha)
{
  PetscFunctionBegin;
  CHKERRQ(MatScale(inA,alpha));
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatScaleUserImpl(Mat,PetscScalar);

static PetscErrorCode MatScaleUserImpl_MPIAIJ(Mat A,PetscScalar aa)
{
  Mat            AA,AB;

  PetscFunctionBegin;
  CHKERRQ(MatMPIAIJGetSeqAIJ(A,&AA,&AB,NULL));
  CHKERRQ(MatScaleUserImpl(AA,aa));
  CHKERRQ(MatScaleUserImpl(AB,aa));
  PetscFunctionReturn(0);
}

/* This routine registers MatScaleUserImpl_SeqAIJ() and
   MatScaleUserImpl_MPIAIJ() as methods providing MatScaleUserImpl()
   functionality for SeqAIJ and MPIAIJ matrix-types */
PetscErrorCode RegisterMatScaleUserImpl(Mat mat)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat), &size));
  if (size == 1) { /* SeqAIJ Matrix */
    CHKERRQ(PetscObjectComposeFunction((PetscObject)mat,"MatScaleUserImpl_C",MatScaleUserImpl_SeqAIJ));
  } else { /* MPIAIJ Matrix */
    Mat AA,AB;
    CHKERRQ(MatMPIAIJGetSeqAIJ(mat,&AA,&AB,NULL));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)mat,"MatScaleUserImpl_C",MatScaleUserImpl_MPIAIJ));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)AA,"MatScaleUserImpl_C",MatScaleUserImpl_SeqAIJ));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)AB,"MatScaleUserImpl_C",MatScaleUserImpl_SeqAIJ));
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
  PetscErrorCode (*f)(Mat,PetscScalar);

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQueryFunction((PetscObject)mat,"MatScaleUserImpl_C",&f));
  if (f) CHKERRQ((*f)(mat,a));
  PetscFunctionReturn(0);
}

/* Main user code that uses MatScaleUserImpl() */

int main(int argc,char **args)
{
  Mat            mat;
  PetscInt       i,j,m = 2,n,Ii,J;
  PetscScalar    v,none = -1.0;
  PetscMPIInt    rank,size;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  /* create the matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&mat));
  CHKERRQ(MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetType(mat,MATAIJ));
  CHKERRQ(MatSetUp(mat));

  /* register user defined MatScaleUser() operation for both SeqAIJ
     and MPIAIJ types */
  CHKERRQ(RegisterMatScaleUserImpl(mat));

  /* assemble the matrix */
  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(mat,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(mat,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(mat,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(mat,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(mat,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  /* check the matrix before and after scaling by -1.0 */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix _before_ MatScaleUserImpl() operation\n"));
  CHKERRQ(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatScaleUserImpl(mat,none));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix _after_ MatScaleUserImpl() operation\n"));
  CHKERRQ(MatView(mat,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2

TEST*/
