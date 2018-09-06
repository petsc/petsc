#include <petsc/private/kspimpl.h>
#include <petsc/private/matimpl.h>
#include <petscksp.h>

typedef struct {
  Mat                        A,Ap,B,C,D;
  KSP                        ksp;
  Vec                        work1,work2;
  MatSchurComplementAinvType ainvtype;
} Mat_SchurComplement;

PETSC_INTERN PetscErrorCode MatCreateVecs_SchurComplement(Mat N, Vec*, Vec*);
PETSC_INTERN PetscErrorCode MatView_SchurComplement(Mat, PetscViewer);
PETSC_INTERN PetscErrorCode MatMultTranspose_SchurComplement(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatMult_SchurComplement(Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SchurComplement(Mat, Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode MatSetFromOptions_SchurComplement(PetscOptionItems*, Mat);
PETSC_INTERN PetscErrorCode MatDestroy_SchurComplement(Mat);
PETSC_INTERN PetscErrorCode MatGetSchurComplement_Basic(Mat, IS, IS, IS, IS, MatReuse, Mat*, MatSchurComplementAinvType, MatReuse, Mat*);

PETSC_EXTERN PetscErrorCode MatCreate_SchurComplement(Mat);
