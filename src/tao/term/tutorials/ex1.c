const char help[] = "Basic TaoTerm usage";

#include <petsctao.h>

int main(int argc, char **argv)
{
  TaoTerm     term;
  PetscViewer viewer;
  PetscBool   view_detail = PETSC_TRUE;
  PetscBool   dup_type    = PETSC_FALSE;
  PetscBool   flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_detail", &view_detail, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-duplicate_type", &dup_type, &flg));
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscViewerSetFromOptions(viewer));
  if (view_detail) PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO_DETAIL));
  else PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));

  PetscCall(TaoTermCreate(PETSC_COMM_WORLD, &term));
  PetscCall(TaoTermSetSolutionSizes(term, PETSC_DECIDE, 10, 1));
  PetscCall(TaoTermSetParametersSizes(term, PETSC_DECIDE, 7, 1));
  PetscCall(PetscObjectSetName((PetscObject)term, "example TaoTerm"));
  PetscCall(TaoTermSetFromOptions(term));
  PetscCall(TaoTermSetUp(term));
  if (dup_type) {
    TaoTermType ttype, ttype2;
    TaoTerm     term2;
    PetscBool   is_same;

    PetscCall(TaoTermDuplicate(term, TAOTERM_DUPLICATE_TYPE, &term2));
    PetscCall(TaoTermGetType(term, &ttype));
    PetscCall(TaoTermGetType(term2, &ttype2));
    PetscCall(PetscStrcmp(ttype, ttype2, &is_same));
    PetscCheck(is_same, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "TaoTermDuplicate not duplicating type correctly");
    PetscCall(TaoTermDestroy(&term2));
  }
  PetscCall(TaoTermView(term, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

  test:
    suffix: 0_from_options
    output_file: output/ex1_0.out
    args: -tao_term_type shell

  test:
    suffix: 1
    args: -tao_term_type callbacks

  test:
    suffix: 1_ascii
    args: -tao_term_type callbacks -view_detail 0

  test:
    suffix: 2
    args: -tao_term_type sum -tao_term_sum_number_terms 2 -term_0_tao_term_type halfl2squared -term_1_tao_term_type l1 -tao_term_sum_term_0_scale 0.5 -tao_term_sum_term_0_mask objective,gradient,hessian

  test:
    suffix: 2_ascii
    args: -tao_term_type sum -tao_term_sum_number_terms 2 -term_0_tao_term_type halfl2squared -term_1_tao_term_type l1 -tao_term_sum_term_0_scale 0.5 -tao_term_sum_term_0_mask objective,gradient,hessian
    args: -view_detail 0

  test:
    suffix: duplicate_type
    output_file: output/ex1_0.out
    args: -tao_term_type shell -duplicate_type 1

TEST*/
