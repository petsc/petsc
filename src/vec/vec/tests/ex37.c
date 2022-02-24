static char help[] = "Nest vector functionality.\n\n";

/*T
   Concepts: vectors^block operators
   Concepts: vectors^setting values
   Concepts: vectors^local access to
   Processors: n
T*/

#include <petscvec.h>

static PetscErrorCode GetISs(Vec vecs[],IS is[])
{
  PetscInt       rstart[2],rend[2];

  PetscFunctionBegin;
  CHKERRQ(VecGetOwnershipRange(vecs[0],&rstart[0],&rend[0]));
  CHKERRQ(VecGetOwnershipRange(vecs[1],&rstart[1],&rend[1]));
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,rend[0]-rstart[0],rstart[0]+rstart[1],1,&is[0]));
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,rend[1]-rstart[1],rend[0]+rstart[1],1,&is[1]));
  PetscFunctionReturn(0);
}

PetscErrorCode test_view(void)
{
  Vec            X, a,b;
  Vec            c,d,e,f;
  Vec            tmp_buf[2];
  IS             tmp_is[2];
  PetscInt       index;
  PetscReal      val;
  PetscInt       list[]={0,1,2};
  PetscScalar    vals[]={0.720032,0.061794,0.0100223};
  PetscBool      explcit = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\n\n============== %s ==============\n", PETSC_FUNCTION_NAME));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &c));
  CHKERRQ(VecSetSizes(c, PETSC_DECIDE, 3));
  CHKERRQ(VecSetFromOptions(c));
  CHKERRQ(VecDuplicate(c, &d));
  CHKERRQ(VecDuplicate(c, &e));
  CHKERRQ(VecDuplicate(c, &f));

  CHKERRQ(VecSet(c, 1.0));
  CHKERRQ(VecSet(d, 2.0));
  CHKERRQ(VecSet(e, 3.0));
  CHKERRQ(VecSetValues(f,3,list,vals,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(f));
  CHKERRQ(VecAssemblyEnd(f));
  CHKERRQ(VecScale(f, 10.0));

  tmp_buf[0] = e;
  tmp_buf[1] = f;
  CHKERRQ(PetscOptionsGetBool(NULL,0,"-explicit_is",&explcit,0));
  CHKERRQ(GetISs(tmp_buf,tmp_is));
  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,explcit ? tmp_is : NULL,tmp_buf,&b));
  CHKERRQ(VecDestroy(&e));
  CHKERRQ(VecDestroy(&f));
  CHKERRQ(ISDestroy(&tmp_is[0]));
  CHKERRQ(ISDestroy(&tmp_is[1]));

  tmp_buf[0] = c;
  tmp_buf[1] = d;
  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&a));
  CHKERRQ(VecDestroy(&c));   CHKERRQ(VecDestroy(&d));

  tmp_buf[0] = a;
  tmp_buf[1] = b;
  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&X));
  CHKERRQ(VecDestroy(&a));

  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));

  CHKERRQ(VecMax(b, &index, &val));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(max-b) = %f : index = %" PetscInt_FMT " \n",(double) val, index));

  CHKERRQ(VecMin(b, &index, &val));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(min-b) = %f : index = %" PetscInt_FMT " \n",(double) val, index));

  CHKERRQ(VecDestroy(&b));

  CHKERRQ(VecMax(X, &index, &val));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(max-X) = %f : index = %" PetscInt_FMT " \n",(double) val, index));
  CHKERRQ(VecMin(X, &index, &val));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(min-X) = %f : index = %" PetscInt_FMT " \n",(double) val, index));

  CHKERRQ(VecView(X, PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&X));
  PetscFunctionReturn(0);
}

#if 0
PetscErrorCode test_vec_ops(void)
{
  Vec            X, a,b;
  Vec            c,d,e,f;
  PetscScalar    val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\n\n============== %s ==============\n",PETSC_FUNCTION_NAME));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &X));
  CHKERRQ(VecSetSizes(X, 2, 2));
  CHKERRQ(VecSetType(X, VECNEST));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &a));
  CHKERRQ(VecSetSizes(a, 2, 2));
  CHKERRQ(VecSetType(a, VECNEST));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &b));
  CHKERRQ(VecSetSizes(b, 2, 2));
  CHKERRQ(VecSetType(b, VECNEST));

  /* assemble X */
  CHKERRQ(VecNestSetSubVec(X, 0, a));
  CHKERRQ(VecNestSetSubVec(X, 1, b));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &c));
  CHKERRQ(VecSetSizes(c, 3, 3));
  CHKERRQ(VecSetType(c, VECSEQ));
  CHKERRQ(VecDuplicate(c, &d));
  CHKERRQ(VecDuplicate(c, &e));
  CHKERRQ(VecDuplicate(c, &f));

  CHKERRQ(VecSet(c, 1.0));
  CHKERRQ(VecSet(d, 2.0));
  CHKERRQ(VecSet(e, 3.0));
  CHKERRQ(VecSet(f, 4.0));

  /* assemble a */
  CHKERRQ(VecNestSetSubVec(a, 0, c));
  CHKERRQ(VecNestSetSubVec(a, 1, d));
  CHKERRQ(VecAssemblyBegin(a));
  CHKERRQ(VecAssemblyEnd(a));

  /* assemble b */
  CHKERRQ(VecNestSetSubVec(b, 0, e));
  CHKERRQ(VecNestSetSubVec(b, 1, f));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));

  CHKERRQ(VecDot(X,X, &val));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X.X = %f \n",(double) val));
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode gen_test_vector(MPI_Comm comm, PetscInt length, PetscInt start_value, PetscInt stride, Vec *_v)
{
  int            size;
  Vec            v;
  PetscInt       i;
  PetscScalar    vx;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(VecCreate(comm, &v));
  CHKERRQ(VecSetSizes(v, PETSC_DECIDE, length));
  if (size == 1) CHKERRQ(VecSetType(v, VECSEQ));
  else CHKERRQ(VecSetType(v, VECMPI));

  for (i=0; i<length; i++) {
    vx   = (PetscScalar)(start_value + i * stride);
    CHKERRQ(VecSetValue(v, i, vx, INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(v));
  CHKERRQ(VecAssemblyEnd(v));

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
  Vec            x1,y1, x2,y2;
  Vec            tmp_buf[2];
  Vec            X, Y;
  PetscReal      real,real2;
  PetscScalar    scalar;
  PetscInt       index;

  PetscFunctionBegin;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\n\n============== %s ==============\n", PETSC_FUNCTION_NAME));

  gen_test_vector(PETSC_COMM_WORLD, 4, 0, 1, &x1);
  gen_test_vector(PETSC_COMM_WORLD, 5, 10, 2, &x2);

  gen_test_vector(PETSC_COMM_WORLD, 4, 4, 3, &y1);
  gen_test_vector(PETSC_COMM_WORLD, 5, 5, 1, &y2);

  tmp_buf[0] = x1;
  tmp_buf[1] = x2;
  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&X));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));
  CHKERRQ(VecDestroy(&x1));
  CHKERRQ(VecDestroy(&x2));

  tmp_buf[0] = y1;
  tmp_buf[1] = y2;
  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&Y));
  CHKERRQ(VecAssemblyBegin(Y));
  CHKERRQ(VecAssemblyEnd(Y));
  CHKERRQ(VecDestroy(&y1));
  CHKERRQ(VecDestroy(&y2));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "VecAXPY \n"));
  CHKERRQ(VecAXPY(Y, 1.0, X)); /* Y <- a X + Y */
  CHKERRQ(VecNestGetSubVec(Y, 0, &y1));
  CHKERRQ(VecNestGetSubVec(Y, 1, &y2));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(1) y1 = \n"));
  CHKERRQ(VecView(y1, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(1) y2 = \n"));
  CHKERRQ(VecView(y2, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDot(X,Y, &scalar));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi \n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar)));

  CHKERRQ(VecDotNorm2(X,Y, &scalar, &real2));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi     norm2(Y) = %lf\n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar), (double)real2));

  CHKERRQ(VecAXPY(Y, 1.0, X)); /* Y <- a X + Y */
  CHKERRQ(VecNestGetSubVec(Y, 0, &y1));
  CHKERRQ(VecNestGetSubVec(Y, 1, &y2));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(2) y1 = \n"));
  CHKERRQ(VecView(y1, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(2) y2 = \n"));
  CHKERRQ(VecView(y2, PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDot(X,Y, &scalar));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi \n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar)));
  CHKERRQ(VecDotNorm2(X,Y, &scalar, &real2));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi     norm2(Y) = %lf\n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar), (double)real2));

  CHKERRQ(VecMax(X, &index, &real));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(max-X) = %f : index = %" PetscInt_FMT " \n",(double) real, index));
  CHKERRQ(VecMin(X, &index, &real));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "(min-X) = %f : index = %" PetscInt_FMT " \n",(double) real, index));

  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args,(char*)0, help);if (ierr) return ierr;
  CHKERRQ(test_view());
  CHKERRQ(test_axpy_dot_max());
  ierr = PetscFinalize();
  return ierr;
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
        requires: kokkos_kernels
        suffix: kokkos
        args: -vec_type kokkos

      test:
        requires: hip
        suffix: hip
        args: -vec_type hip

TEST*/
