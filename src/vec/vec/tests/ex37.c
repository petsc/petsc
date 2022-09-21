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
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
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

  gen_test_vector(PETSC_COMM_WORLD, 4, 0, 1, &x1);
  gen_test_vector(PETSC_COMM_WORLD, 5, 10, 2, &x2);

  gen_test_vector(PETSC_COMM_WORLD, 4, 4, 3, &y1);
  gen_test_vector(PETSC_COMM_WORLD, 5, 5, 1, &y2);

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
  PetscFunctionReturn(0);
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
