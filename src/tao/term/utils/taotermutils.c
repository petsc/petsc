#include <petsc/private/taoimpl.h>

PETSC_INTERN PetscErrorCode VecIfNotCongruentGetSameLayoutVec(Vec a, Vec *b)
{
  PetscFunctionBegin;
  if (*b == NULL) {
    PetscCall(VecDuplicate(a, b));
  } else {
    PetscLayout layout_a, layout_b;
    PetscBool   is_same;

    PetscCall(VecGetLayout(a, &layout_a));
    PetscCall(VecGetLayout(*b, &layout_b));
    PetscCall(PetscLayoutCompare(layout_a, layout_b, &is_same));
    if (is_same == PETSC_FALSE) {
      PetscCall(VecDestroy(b));
      PetscCall(VecDuplicate(a, b));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateHessianMatricesDefault_H_Internal(TaoTerm term, Mat *H, Mat *Hpre, PetscBool Hpre_is_H, MatType H_mattype)
{
  Mat       Htemp;
  PetscBool is_mffd = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(H_mattype, MATMFFD, &is_mffd));
  if (is_mffd) {
    PetscCall(TaoTermCreateHessianMFFD(term, &Htemp));
  } else {
    PetscLayout sol_layout;
    VecType     sol_vec_type;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)term), &Htemp));
    PetscCall(TaoTermGetSolutionLayout(term, &sol_layout));
    PetscCall(MatSetLayouts(Htemp, sol_layout, sol_layout));
    PetscCall(TaoTermGetSolutionVecType(term, &sol_vec_type));
    if (H_mattype) PetscCall(MatSetType(Htemp, H_mattype));
    else PetscCall(MatSetVecType(Htemp, sol_vec_type));
    PetscCall(MatSetOption(Htemp, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(Htemp, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  }

  if (H) {
    PetscCall(PetscObjectReference((PetscObject)Htemp));
    *H = Htemp;
  }
  if (Hpre && Hpre_is_H) {
    PetscCall(PetscObjectReference((PetscObject)Htemp));
    *Hpre = Htemp;
  }
  PetscCall(MatDestroy(&Htemp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermCreateHessianMatricesDefault_Hpre_Internal(TaoTerm term, Mat *H, Mat *Hpre, PetscBool Hpre_is_H, MatType Hpre_mattype)
{
  Mat       Hpretemp;
  PetscBool is_mffd = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(Hpre_mattype, MATMFFD, &is_mffd));
  if (is_mffd) {
    PetscCall(TaoTermCreateHessianMFFD(term, &Hpretemp));
  } else {
    PetscLayout sol_layout;
    VecType     sol_vec_type;

    PetscCall(MatCreate(PetscObjectComm((PetscObject)term), &Hpretemp));
    PetscCall(TaoTermGetSolutionLayout(term, &sol_layout));
    PetscCall(MatSetLayouts(Hpretemp, sol_layout, sol_layout));
    PetscCall(TaoTermGetSolutionVecType(term, &sol_vec_type));
    if (Hpre_mattype) PetscCall(MatSetType(Hpretemp, Hpre_mattype));
    else PetscCall(MatSetVecType(Hpretemp, sol_vec_type));
    PetscCall(MatSetOption(Hpretemp, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(Hpretemp, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  }
  *Hpre = Hpretemp;
  PetscFunctionReturn(PETSC_SUCCESS);
}
