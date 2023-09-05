const char help[] = "Test construction of a mixed form degree dual space";

#include <petscdualspace.h>
#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM              K;
  PetscDualSpace  h_1, h_div, list[2], joint;
  PetscQuadrature quad;
  Mat             mat;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DM_POLYTOPE_QUADRILATERAL, &K));
  PetscCall(PetscDualSpaceCreate(PETSC_COMM_SELF, &h_1));
  PetscCall(PetscObjectSetName((PetscObject)h_1, "H1"));
  PetscCall(PetscDualSpaceSetType(h_1, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscDualSpaceSetDM(h_1, K));
  PetscCall(PetscDualSpaceSetNumComponents(h_1, 2));
  PetscCall(PetscDualSpaceLagrangeSetTensor(h_1, PETSC_TRUE));
  PetscCall(PetscDualSpaceSetFormDegree(h_1, 0));
  PetscCall(PetscDualSpaceSetOrder(h_1, 1));
  PetscCall(PetscDualSpaceSetUp(h_1));
  PetscCall(PetscDualSpaceCreate(PETSC_COMM_SELF, &h_div));
  PetscCall(PetscObjectSetName((PetscObject)h_div, "Hdiv"));
  PetscCall(PetscDualSpaceSetType(h_div, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscDualSpaceSetDM(h_div, K));
  PetscCall(PetscDualSpaceSetNumComponents(h_div, 2));
  PetscCall(PetscDualSpaceLagrangeSetTensor(h_div, PETSC_TRUE));
  PetscCall(PetscDualSpaceSetFormDegree(h_div, -1));
  PetscCall(PetscDualSpaceLagrangeSetTrimmed(h_div, PETSC_TRUE));
  PetscCall(PetscDualSpaceSetOrder(h_div, 1));
  PetscCall(PetscDualSpaceSetUp(h_div));
  list[0] = h_1;
  list[1] = h_div;
  PetscCall(PetscDualSpaceCreateSum(2, list, PETSC_FALSE, &joint));
  PetscCall(PetscDualSpaceSetUp(joint));
  PetscCall(PetscDualSpaceView(joint, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscDualSpaceGetAllData(joint, &quad, &mat));
  PetscCall(PetscQuadratureView(quad, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscObjectSetName((PetscObject)mat, "Joint dual space projection matrix"));
  PetscCall(MatView(mat, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscDualSpaceDestroy(&joint));
  PetscCall(PetscDualSpaceDestroy(&h_div));
  PetscCall(PetscDualSpaceDestroy(&h_1));
  PetscCall(DMDestroy(&K));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
