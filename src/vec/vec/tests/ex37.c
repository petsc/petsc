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
  PetscErrorCode ierr;
  PetscInt       rstart[2],rend[2];

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(vecs[0],&rstart[0],&rend[0]);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(vecs[1],&rstart[1],&rend[1]);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend[0]-rstart[0],rstart[0]+rstart[1],1,&is[0]);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend[1]-rstart[1],rend[0]+rstart[1],1,&is[1]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      explcit = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\n\n============== %s ==============\n", PETSC_FUNCTION_NAME);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &c);CHKERRQ(ierr);
  ierr = VecSetSizes(c, PETSC_DECIDE, 3);CHKERRQ(ierr);
  ierr = VecSetFromOptions(c);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &d);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &e);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &f);CHKERRQ(ierr);

  ierr = VecSet(c, 1.0);CHKERRQ(ierr);
  ierr = VecSet(d, 2.0);CHKERRQ(ierr);
  ierr = VecSet(e, 3.0);CHKERRQ(ierr);
  ierr = VecSetValues(f,3,list,vals,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  ierr = VecScale(f, 10.0);CHKERRQ(ierr);

  tmp_buf[0] = e;
  tmp_buf[1] = f;
  ierr       = PetscOptionsGetBool(NULL,0,"-explicit_is",&explcit,0);CHKERRQ(ierr);
  ierr       = GetISs(tmp_buf,tmp_is);CHKERRQ(ierr);
  ierr       = VecCreateNest(PETSC_COMM_WORLD,2,explcit ? tmp_is : NULL,tmp_buf,&b);CHKERRQ(ierr);
  ierr       = VecDestroy(&e);CHKERRQ(ierr);
  ierr       = VecDestroy(&f);CHKERRQ(ierr);
  ierr       = ISDestroy(&tmp_is[0]);CHKERRQ(ierr);
  ierr       = ISDestroy(&tmp_is[1]);CHKERRQ(ierr);

  tmp_buf[0] = c;
  tmp_buf[1] = d;
  ierr       = VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&a);CHKERRQ(ierr);
  ierr       = VecDestroy(&c);CHKERRQ(ierr);   ierr = VecDestroy(&d);CHKERRQ(ierr);

  tmp_buf[0] = a;
  tmp_buf[1] = b;
  ierr       = VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&X);CHKERRQ(ierr);
  ierr       = VecDestroy(&a);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);

  ierr = VecMax(b, &index, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(max-b) = %f : index = %D \n",(double) val, index);CHKERRQ(ierr);

  ierr = VecMin(b, &index, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(min-b) = %f : index = %D \n",(double) val, index);CHKERRQ(ierr);

  ierr = VecDestroy(&b);CHKERRQ(ierr);

  ierr = VecMax(X, &index, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(max-X) = %f : index = %D \n",(double) val, index);CHKERRQ(ierr);
  ierr = VecMin(X, &index, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(min-X) = %f : index = %D \n",(double) val, index);CHKERRQ(ierr);

  ierr = VecView(X, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
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
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\n\n============== %s ==============\n",PETSC_FUNCTION_NAME);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &X);CHKERRQ(ierr);
  ierr = VecSetSizes(X, 2, 2);CHKERRQ(ierr);
  ierr = VecSetType(X, VECNEST);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &a);CHKERRQ(ierr);
  ierr = VecSetSizes(a, 2, 2);CHKERRQ(ierr);
  ierr = VecSetType(a, VECNEST);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &b);CHKERRQ(ierr);
  ierr = VecSetSizes(b, 2, 2);CHKERRQ(ierr);
  ierr = VecSetType(b, VECNEST);CHKERRQ(ierr);

  /* assemble X */
  ierr = VecNestSetSubVec(X, 0, a);CHKERRQ(ierr);
  ierr = VecNestSetSubVec(X, 1, b);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &c);CHKERRQ(ierr);
  ierr = VecSetSizes(c, 3, 3);CHKERRQ(ierr);
  ierr = VecSetType(c, VECSEQ);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &d);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &e);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &f);CHKERRQ(ierr);

  ierr = VecSet(c, 1.0);CHKERRQ(ierr);
  ierr = VecSet(d, 2.0);CHKERRQ(ierr);
  ierr = VecSet(e, 3.0);CHKERRQ(ierr);
  ierr = VecSet(f, 4.0);CHKERRQ(ierr);

  /* assemble a */
  ierr = VecNestSetSubVec(a, 0, c);CHKERRQ(ierr);
  ierr = VecNestSetSubVec(a, 1, d);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(a);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(a);CHKERRQ(ierr);

  /* assemble b */
  ierr = VecNestSetSubVec(b, 0, e);CHKERRQ(ierr);
  ierr = VecNestSetSubVec(b, 1, f);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  ierr = VecDot(X,X, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "X.X = %f \n",(double) val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode gen_test_vector(MPI_Comm comm, PetscInt length, PetscInt start_value, PetscInt stride, Vec *_v)
{
  int            size;
  Vec            v;
  PetscInt       i;
  PetscScalar    vx;
  PetscErrorCode ierr;

  MPI_Comm_size(comm, &size);

  ierr = VecCreate(comm, &v);CHKERRQ(ierr);
  ierr = VecSetSizes(v, PETSC_DECIDE, length);CHKERRQ(ierr);
  if (size == 1) { ierr = VecSetType(v, VECSEQ);CHKERRQ(ierr); }
  else { ierr = VecSetType(v, VECMPI);CHKERRQ(ierr); }

  for (i=0; i<length; i++) {
    vx   = (PetscScalar)(start_value + i * stride);
    ierr = VecSetValue(v, i, vx, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);CHKERRQ(ierr);

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\n\n============== %s ==============\n", PETSC_FUNCTION_NAME);CHKERRQ(ierr);

  gen_test_vector(PETSC_COMM_WORLD, 4, 0, 1, &x1);
  gen_test_vector(PETSC_COMM_WORLD, 5, 10, 2, &x2);

  gen_test_vector(PETSC_COMM_WORLD, 4, 4, 3, &y1);
  gen_test_vector(PETSC_COMM_WORLD, 5, 5, 1, &y2);

  tmp_buf[0] = x1;
  tmp_buf[1] = x2;
  ierr       = VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&X);CHKERRQ(ierr);
  ierr       = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr       = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr       = VecDestroy(&x1);CHKERRQ(ierr);
  ierr       = VecDestroy(&x2);CHKERRQ(ierr);


  tmp_buf[0] = y1;
  tmp_buf[1] = y2;
  ierr       = VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&Y);CHKERRQ(ierr);
  ierr       = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr       = VecAssemblyEnd(Y);CHKERRQ(ierr);
  ierr       = VecDestroy(&y1);CHKERRQ(ierr);
  ierr       = VecDestroy(&y2);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "VecAXPY \n");CHKERRQ(ierr);
  ierr = VecAXPY(Y, 1.0, X);CHKERRQ(ierr); /* Y <- a X + Y */
  ierr = VecNestGetSubVec(Y, 0, &y1);CHKERRQ(ierr);
  ierr = VecNestGetSubVec(Y, 1, &y2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(1) y1 = \n");CHKERRQ(ierr);
  ierr = VecView(y1, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(1) y2 = \n");CHKERRQ(ierr);
  ierr = VecView(y2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDot(X,Y, &scalar);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi \n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar));CHKERRQ(ierr);

  ierr = VecDotNorm2(X,Y, &scalar, &real2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi     norm2(Y) = %lf\n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar), (double)real2);CHKERRQ(ierr);


  ierr = VecAXPY(Y, 1.0, X);CHKERRQ(ierr); /* Y <- a X + Y */
  ierr = VecNestGetSubVec(Y, 0, &y1);CHKERRQ(ierr);
  ierr = VecNestGetSubVec(Y, 1, &y2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(2) y1 = \n");CHKERRQ(ierr);
  ierr = VecView(y1, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(2) y2 = \n");CHKERRQ(ierr);
  ierr = VecView(y2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDot(X,Y, &scalar);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi \n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar));CHKERRQ(ierr);
  ierr = VecDotNorm2(X,Y, &scalar, &real2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "X.Y = %lf + %lfi     norm2(Y) = %lf\n", (double)PetscRealPart(scalar), (double)PetscImaginaryPart(scalar), (double)real2);CHKERRQ(ierr);


  ierr = VecMax(X, &index, &real);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(max-X) = %f : index = %D \n",(double) real, index);CHKERRQ(ierr);
  ierr = VecMin(X, &index, &real);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "(min-X) = %f : index = %D \n",(double) real, index);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


int main(int argc, char **args)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args,(char*)0, help);if (ierr) return ierr;
  ierr = test_view();CHKERRQ(ierr);
  ierr = test_axpy_dot_max();CHKERRQ(ierr);
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

   test:
      suffix: 4
      nsize: 2
      args: -explicit_is 1

TEST*/
