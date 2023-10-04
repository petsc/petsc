static char help[] = "Nest vector functionality.\n\n";

#include <petscvec.h>

static PetscErrorCode GetISs(Vec vecs[], IS is[], PetscBool inv)
{
  PetscInt rstart[2], rend[2];

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(vecs[0], &rstart[0], &rend[0]));
  PetscCall(VecGetOwnershipRange(vecs[1], &rstart[1], &rend[1]));
  if (!inv) {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, rend[0] - rstart[0], rstart[0] + rstart[1], 1, &is[0]));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, rend[1] - rstart[1], rend[0] + rstart[1], 1, &is[1]));
  } else {
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, rend[0] - rstart[0], rend[0] + rend[1] - 1, -1, &is[0]));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, rend[1] - rstart[1], rstart[0] + rend[1] - 1, -1, &is[1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode convert_from_nest(Vec X, Vec *Y)
{
  const PetscScalar *v;
  PetscInt           rstart, n, N;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetSize(X, &N));
  PetscCall(VecGetOwnershipRange(X, &rstart, NULL));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)X), Y));
  PetscCall(VecSetSizes(*Y, n, N));
  PetscCall(VecSetType(*Y, VECSTANDARD)); // We always use a CPU only version
  PetscCall(VecGetArrayRead(X, &v));
  for (PetscInt i = 0; i < n; i++) PetscCall(VecSetValue(*Y, rstart + i, v[i], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X, &v));
  PetscCall(VecAssemblyBegin(*Y));
  PetscCall(VecAssemblyEnd(*Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test_view(void)
{
  Vec          X, lX, a, b;
  Vec          c, d, e, f;
  Vec          tmp_buf[2];
  IS           tmp_is[2];
  PetscInt     index, n;
  PetscReal    val;
  PetscInt     list[] = {0, 1, 2};
  PetscScalar  vals[] = {0.5, 0.25, 0.125};
  PetscScalar *x, *lx;
  PetscBool    explcit = PETSC_FALSE, inv = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n============== %s ==============\n", PETSC_FUNCTION_NAME));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &c));
  PetscCall(VecSetSizes(c, PETSC_DECIDE, 3));
  PetscCall(VecSetFromOptions(c));
  PetscCall(VecDuplicate(c, &d));
  PetscCall(VecDuplicate(c, &e));
  PetscCall(VecDuplicate(c, &f));

  PetscCall(VecSet(c, 1.0));
  PetscCall(VecSet(d, 2.0));
  PetscCall(VecSet(e, 3.0));
  PetscCall(VecSetValues(f, 3, list, vals, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscCall(VecScale(f, 10.0));

  tmp_buf[0] = e;
  tmp_buf[1] = f;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-explicit_is", &explcit, 0));
  PetscCall(GetISs(tmp_buf, tmp_is, PETSC_FALSE));
  PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, explcit ? tmp_is : NULL, tmp_buf, &b));
  PetscCall(VecDestroy(&e));
  PetscCall(VecDestroy(&f));
  PetscCall(ISDestroy(&tmp_is[0]));
  PetscCall(ISDestroy(&tmp_is[1]));

  tmp_buf[0] = c;
  tmp_buf[1] = d;
  PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, NULL, tmp_buf, &a));
  PetscCall(VecDestroy(&c));
  PetscCall(VecDestroy(&d));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-inv", &inv, 0));
  tmp_buf[0] = a;
  tmp_buf[1] = b;
  if (inv) {
    PetscCall(GetISs(tmp_buf, tmp_is, inv));
    PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, tmp_is, tmp_buf, &X));
    PetscCall(ISDestroy(&tmp_is[0]));
    PetscCall(ISDestroy(&tmp_is[1]));
  } else {
    PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, NULL, tmp_buf, &X));
  }
  PetscCall(VecDestroy(&a));

  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  PetscCall(VecMax(b, &index, &val));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(max-b) = %f : index = %" PetscInt_FMT " \n", (double)val, index));

  PetscCall(VecMin(b, &index, &val));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(min-b) = %f : index = %" PetscInt_FMT " \n", (double)val, index));

  PetscCall(VecDestroy(&b));

  PetscCall(VecMax(X, &index, &val));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(max-X) = %f : index = %" PetscInt_FMT " \n", (double)val, index));
  PetscCall(VecMin(X, &index, &val));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(min-X) = %f : index = %" PetscInt_FMT " \n", (double)val, index));

  PetscCall(VecView(X, PETSC_VIEWER_STDOUT_WORLD));

  Vec X2, A, R, E, vX, vX2, vA, vR, vE;
  PetscCall(convert_from_nest(X, &vX));
  PetscCall(VecDuplicate(X, &X2));
  PetscCall(VecDuplicate(X, &A));
  PetscCall(VecDuplicate(X, &R));
  PetscCall(VecDuplicate(X, &E));
  PetscCall(VecSetRandom(A, NULL));
  PetscCall(VecSetRandom(R, NULL));
  PetscCall(VecSetRandom(E, NULL));
  PetscCall(convert_from_nest(A, &vA));
  PetscCall(convert_from_nest(R, &vR));
  PetscCall(convert_from_nest(E, &vE));
  PetscCall(VecCopy(X, X2));
  PetscCall(VecDuplicate(vX, &vX2));
  PetscCall(VecCopy(vX, vX2));
  PetscCall(VecScale(X2, 2.0));
  PetscCall(VecScale(vX2, 2.0));
  for (int nt = 0; nt < 2; nt++) {
    NormType norm = nt ? NORM_INFINITY : NORM_2;
    for (int e = 0; e < 2; e++) {
      for (int a = 0; a < 2; a++) {
        for (int r = 0; r < 2; r++) {
          PetscReal vn, vna, vnr, nn, nna, nnr;
          PetscInt  vn_loc, vna_loc, vnr_loc, nn_loc, nna_loc, nnr_loc;

          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Testing Wnorms %s: E? %d A? %d R? %d\n", norm == NORM_2 ? "2" : "inf", e, a, r));
          PetscCall(VecErrorWeightedNorms(vX, vX2, e ? vE : NULL, norm, 0.5, a ? vA : NULL, 0.5, r ? vR : NULL, 0.0, &vn, &vn_loc, &vna, &vna_loc, &vnr, &vnr_loc));
          PetscCall(VecErrorWeightedNorms(X, X2, e ? E : NULL, norm, 0.5, a ? A : NULL, 0.5, r ? R : NULL, 0.0, &nn, &nn_loc, &nna, &nna_loc, &nnr, &nnr_loc));
          if (vn_loc != nn_loc) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   error with total norm loc: %" PetscInt_FMT " %" PetscInt_FMT "\n", vn_loc, nn_loc));
          if (vna_loc != nna_loc) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   error with absolute norm loc: %" PetscInt_FMT " %" PetscInt_FMT "\n", vna_loc, nna_loc));
          if (vnr_loc != nnr_loc) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   error with relative norm loc: %" PetscInt_FMT " %" PetscInt_FMT "\n", vnr_loc, nnr_loc));
          if (!PetscIsCloseAtTol(vna, nna, 0, PETSC_SQRT_MACHINE_EPSILON)) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   error with absolute norm: %1.16e %1.16e diff %1.16e\n", (double)vna, (double)nna, (double)(vna - nna)));
          if (!PetscIsCloseAtTol(vnr, nnr, 0, PETSC_SQRT_MACHINE_EPSILON)) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   error with relative norm: %1.16e %1.16e diff %1.16e\n", (double)vnr, (double)nnr, (double)(vnr - nnr)));
        }
      }
    }
  }
  PetscCall(VecDestroy(&X2));
  PetscCall(VecDestroy(&A));
  PetscCall(VecDestroy(&R));
  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&vX));
  PetscCall(VecDestroy(&vX2));
  PetscCall(VecDestroy(&vA));
  PetscCall(VecDestroy(&vR));
  PetscCall(VecDestroy(&vE));

  PetscCall(VecCreateLocalVector(X, &lX));
  PetscCall(VecGetLocalVectorRead(X, lX));
  PetscCall(VecGetLocalSize(lX, &n));
  PetscCall(VecGetArrayRead(lX, (const PetscScalar **)&lx));
  PetscCall(VecGetArrayRead(X, (const PetscScalar **)&x));
  for (PetscInt i = 0; i < n; i++) PetscCheck(lx[i] == x[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid data");
  PetscCall(VecRestoreArrayRead(X, (const PetscScalar **)&x));
  PetscCall(VecRestoreArrayRead(lX, (const PetscScalar **)&lx));
  PetscCall(VecRestoreLocalVectorRead(X, lX));

  PetscCall(VecDestroy(&lX));
  PetscCall(VecDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if 0
PetscErrorCode test_vec_ops(void)
{
  Vec            X, a,b;
  Vec            c,d,e,f;
  PetscScalar    val;

  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n============== %s ==============\n",PETSC_FUNCTION_NAME));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, 2, 2));
  PetscCall(VecSetType(X, VECNEST));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &a));
  PetscCall(VecSetSizes(a, 2, 2));
  PetscCall(VecSetType(a, VECNEST));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
  PetscCall(VecSetSizes(b, 2, 2));
  PetscCall(VecSetType(b, VECNEST));

  /* assemble X */
  PetscCall(VecNestSetSubVec(X, 0, a));
  PetscCall(VecNestSetSubVec(X, 1, b));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &c));
  PetscCall(VecSetSizes(c, 3, 3));
  PetscCall(VecSetType(c, VECSEQ));
  PetscCall(VecDuplicate(c, &d));
  PetscCall(VecDuplicate(c, &e));
  PetscCall(VecDuplicate(c, &f));

  PetscCall(VecSet(c, 1.0));
  PetscCall(VecSet(d, 2.0));
  PetscCall(VecSet(e, 3.0));
  PetscCall(VecSet(f, 4.0));

  /* assemble a */
  PetscCall(VecNestSetSubVec(a, 0, c));
  PetscCall(VecNestSetSubVec(a, 1, d));
  PetscCall(VecAssemblyBegin(a));
  PetscCall(VecAssemblyEnd(a));

  /* assemble b */
  PetscCall(VecNestSetSubVec(b, 0, e));
  PetscCall(VecNestSetSubVec(b, 1, f));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  PetscCall(VecDot(X,X, &val));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X.X = %f \n",(double) val));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

PetscErrorCode gen_test_vector(MPI_Comm comm, PetscInt length, PetscInt start_value, PetscInt stride, Vec *_v)
{
  PetscMPIInt size;
  Vec         v;
  PetscInt    i;
  PetscScalar vx;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(VecCreate(comm, &v));
  PetscCall(VecSetSizes(v, PETSC_DECIDE, length));
  if (size == 1) PetscCall(VecSetType(v, VECSEQ));
  else PetscCall(VecSetType(v, VECMPI));

  for (i = 0; i < length; i++) {
    vx = (PetscScalar)(start_value + i * stride);
    PetscCall(VecSetValue(v, i, vx, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));

  *_v = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
X = ([0,1,2,3], [10,12,14,16,18])
Y = ([4,7,10,13], [5,6,7,8,9])

Y = aX + y = ([4,8,12,16], (15,18,21,24,27])
Y = aX + y = ([4,9,14,19], (25,30,35,40,45])

*/
PetscErrorCode test_axpy_dot_max(void)
{
  Vec         x1, y1, x2, y2;
  Vec         tmp_buf[2];
  Vec         X, Y;
  PetscReal   real, real2;
  PetscScalar scalar;
  PetscInt    index;

  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n============== %s ==============\n", PETSC_FUNCTION_NAME));

  PetscCall(gen_test_vector(PETSC_COMM_WORLD, 4, 0, 1, &x1));
  PetscCall(gen_test_vector(PETSC_COMM_WORLD, 5, 10, 2, &x2));

  PetscCall(gen_test_vector(PETSC_COMM_WORLD, 4, 4, 3, &y1));
  PetscCall(gen_test_vector(PETSC_COMM_WORLD, 5, 5, 1, &y2));

  tmp_buf[0] = x1;
  tmp_buf[1] = x2;
  PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, NULL, tmp_buf, &X));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));

  tmp_buf[0] = y1;
  tmp_buf[1] = y2;
  PetscCall(VecCreateNest(PETSC_COMM_WORLD, 2, NULL, tmp_buf, &Y));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscCall(VecDestroy(&y1));
  PetscCall(VecDestroy(&y2));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "VecAXPY \n"));
  PetscCall(VecAXPY(Y, 1.0, X)); /* Y <- a X + Y */
  PetscCall(VecNestGetSubVec(Y, 0, &y1));
  PetscCall(VecNestGetSubVec(Y, 1, &y2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(1) y1 = \n"));
  PetscCall(VecView(y1, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(1) y2 = \n"));
  PetscCall(VecView(y2, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDot(X, Y, &scalar));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi \n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar)));

  PetscCall(VecDotNorm2(X, Y, &scalar, &real2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi     norm2(Y) = %lf\n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar), (double)real2));

  PetscCall(VecAXPY(Y, 1.0, X)); /* Y <- a X + Y */
  PetscCall(VecNestGetSubVec(Y, 0, &y1));
  PetscCall(VecNestGetSubVec(Y, 1, &y2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(2) y1 = \n"));
  PetscCall(VecView(y1, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(2) y2 = \n"));
  PetscCall(VecView(y2, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDot(X, Y, &scalar));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi \n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar)));
  PetscCall(VecDotNorm2(X, Y, &scalar, &real2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi     norm2(Y) = %lf\n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar), (double)real2));

  PetscCall(VecMax(X, &index, &real));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(max-X) = %f : index = %" PetscInt_FMT " \n", (double)real, index));
  PetscCall(VecMin(X, &index, &real));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(min-X) = %f : index = %" PetscInt_FMT " \n", (double)real, index));

  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(test_view());
  PetscCall(test_axpy_dot_max());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -explicit_is 0

   test:
      suffix: 2
      args: -explicit_is 1
      output_file: output/ex37_1.out

   test:
      suffix: 3
      nsize: 2
      args: -explicit_is 0

   testset:
      nsize: 2
      args: -explicit_is 1
      output_file: output/ex37_4.out
      filter: grep -v -e "type: mpi" -e "type=mpi"

      test:
        suffix: 4

      test:
        requires: cuda
        suffix: 4_cuda
        args: -vec_type cuda

      test:
        requires: kokkos_kernels
        suffix: 4_kokkos
        args: -vec_type kokkos

      test:
        requires: hip
        suffix: 4_hip
        args: -vec_type hip

   testset:
      nsize: 2
      args: -explicit_is 1 -inv
      output_file: output/ex37_5.out
      filter: grep -v -e "type: mpi" -e "type=mpi"

      test:
        suffix: 5

      test:
        requires: cuda
        suffix: 5_cuda
        args: -vec_type cuda

      test:
        requires: kokkos_kernels
        suffix: 5_kokkos
        args: -vec_type kokkos

      test:
        requires: hip
        suffix: 5_hip
        args: -vec_type hip

TEST*/
