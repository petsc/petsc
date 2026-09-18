static char help[] = "Test MatNullSpaceView()/MatNullSpaceLoad() with a binary viewer.\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  MatNullSpace nsp, loaded;
  PetscViewer  viewer;
  Vec          v;
  const Vec   *vecs;
  PetscInt     rstart, rend, n;
  PetscBool    has_cnst, equal;
  PetscScalar  value;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 2, &v));
  PetscCall(VecGetOwnershipRange(v, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    value = i ? -1.0 : 1.0;
    PetscCall(VecSetValue(v, i, value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscCall(VecNormalize(v, NULL));
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 1, &v, &nsp));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "nullspace.dat", FILE_MODE_WRITE, &viewer));
  PetscCall(MatNullSpaceView(nsp, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "nullspace.dat", FILE_MODE_READ, &viewer));
  PetscCall(MatNullSpaceLoad(viewer, &loaded));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatNullSpaceGetVecs(loaded, &has_cnst, &n, &vecs));
  PetscCheck(has_cnst, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Loaded null space does not contain the constant vector");
  PetscCheck(n == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Loaded null space contains %" PetscInt_FMT " vectors instead of 1", n);
  PetscCall(VecEqual(v, vecs[0], &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Loaded null space vector differs from the original");

  PetscCall(MatNullSpaceDestroy(&loaded));
  PetscCall(MatNullSpaceDestroy(&nsp));
  PetscCall(VecDestroy(&v));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    temporaries: nullspace.dat nullspace.dat.info

    test:
      suffix: 1
      nsize: {{1 2}}
      args: -viewer_binary_skip_header {{0 1}}

TEST*/
