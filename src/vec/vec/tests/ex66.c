const char help[] = "Test VecPointwiseSign()";

#include <petscvec.h>

int main(int argc, char **argv)
{
  MPI_Comm    comm;
  PetscRandom rand;
  PetscInt    n = 12;
  Vec         x, y, z, test;
  VecSignMode types[] = {VEC_SIGN_ZERO_TO_ZERO, VEC_SIGN_ZERO_TO_SIGNED_ZERO, VEC_SIGN_ZERO_TO_SIGNED_UNIT};

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetInterval(rand, 0.5, 1.5));
  PetscCall(VecCreate(comm, &x));
  PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecDuplicate(x, &z));
  PetscCall(VecDuplicate(x, &test));
  {
    PetscScalar *_x;
    PetscCall(VecGetArrayWrite(x, &_x));
    for (PetscInt i = 0; i < n; i++) {
      PetscReal r;

      PetscCall(PetscRandomGetValueReal(rand, &r));
      switch (i % 4) {
      case 0:
        _x[i] = -r;
        break;
      case 1:
        _x[i] = -0.0;
        break;
      case 2:
        _x[i] = 0.0;
        break;
      default:
        _x[i] = r;
        break;
      }
    }
    PetscCall(VecRestoreArrayWrite(x, &_x));
  }
  for (size_t t = 0; t < PETSC_STATIC_ARRAY_LENGTH(types); t++) {
    PetscScalar *_test;
    VecSignMode  sign_type = types[t];

    PetscCall(VecGetArrayWrite(test, &_test));
    for (PetscInt i = 0; i < n; i++) {
      switch (i % 4) {
      case 0:
        _test[i] = -1.0;
        break;
      case 1:
        switch (sign_type) {
        case VEC_SIGN_ZERO_TO_ZERO:
          _test[i] = 0.0;
          break;
        case VEC_SIGN_ZERO_TO_SIGNED_ZERO:
          _test[i] = -0.0;
          break;
        case VEC_SIGN_ZERO_TO_SIGNED_UNIT:
          _test[i] = -1.0;
          break;
        }
        break;
      case 2:
        switch (sign_type) {
        case VEC_SIGN_ZERO_TO_ZERO:
          _test[i] = 0.0;
          break;
        case VEC_SIGN_ZERO_TO_SIGNED_ZERO:
          _test[i] = 0.0;
          break;
        case VEC_SIGN_ZERO_TO_SIGNED_UNIT:
          _test[i] = 1.0;
          break;
        }
        break;
      default:
        _test[i] = 1.0;
        break;
      }
    }
    PetscCall(VecRestoreArrayWrite(test, &_test));
    for (PetscInt j = 0; j < 2; j++) {
      Vec                out = j ? z : y;
      const PetscScalar *_out;
      const PetscScalar *_test;

      PetscCall(VecCopy(x, y));
      PetscCall(VecPointwiseSign(out, y, sign_type));
      PetscCall(VecGetArrayRead(out, &_out));
      PetscCall(VecGetArrayRead(test, &_test));
      for (PetscInt i = 0; i < n; i++) {
        PetscScalar _o = _out[i];
        PetscScalar _t = _test[i];

        PetscCheck(PetscImaginaryPart(_o) == 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nonzero imaginary part");
        PetscCheck(PetscAbsScalar(_o) == PetscAbsScalar(_t), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected magnitude");
        PetscCheck(PetscCopysignReal(1.0, PetscRealPart(_o)) == PetscCopysignReal(1.0, PetscRealPart(_t)), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected sign");
      }
      PetscCall(VecRestoreArrayRead(test, &_test));
      PetscCall(VecRestoreArrayRead(out, &_out));
    }
  }
  PetscCall(VecDestroy(&test));
  PetscCall(VecDestroy(&z));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: {{1 2}}
    suffix: 0
    output_file: output/empty.out

  test:
    requires: cuda
    nsize: {{1 2}}
    suffix: cuda
    output_file: output/empty.out
    args: -vec_type cuda

  test:
    requires: hip
    nsize: {{1 2}}
    suffix: hip
    output_file: output/empty.out
    args: -vec_type hip

TEST*/
