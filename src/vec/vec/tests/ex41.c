static char help[] = "Nest vector set subvector functionality.\n\n";

#include <petscvec.h>

PetscErrorCode test_vec_ops(void)
{
  Vec            X,Y,a,b;
  Vec            c,d,e,f,g,h;
  PetscScalar    val;
  PetscInt       tmp_ind[2];
  Vec            tmp_buf[2];

  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "============== %s ==============\n",PETSC_FUNCTION_NAME));

  /* create 4 worker vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &c));
  PetscCall(VecSetSizes(c, PETSC_DECIDE, 4));
  PetscCall(VecSetType(c, VECMPI));
  PetscCall(VecDuplicate(c, &d));
  PetscCall(VecDuplicate(c, &e));
  PetscCall(VecDuplicate(c, &f));

  /* create two more workers of different sizes */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &g));
  PetscCall(VecSetSizes(g, PETSC_DECIDE, 6));
  PetscCall(VecSetType(g, VECMPI));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &h));
  PetscCall(VecSetSizes(h, PETSC_DECIDE, 8));
  PetscCall(VecSetType(h, VECMPI));

  /* set the 6 vectors to some numbers */
  PetscCall(VecSet(c, 1.0));
  PetscCall(VecSet(d, 2.0));
  PetscCall(VecSet(e, 3.0));
  PetscCall(VecSet(f, 4.0));
  PetscCall(VecSet(g, 5.0));
  PetscCall(VecSet(h, 6.0));

  /* assemble a */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "a = [c d] \n"));
  tmp_buf[0] = c; tmp_buf[1] = d;

  PetscCall(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&a));
  PetscCall(VecView(a,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "a = [d c] \n"));
  PetscCall(VecNestSetSubVec(a, 1, c));
  PetscCall(VecNestSetSubVec(a, 0, d));
  PetscCall(VecAssemblyBegin(a));
  PetscCall(VecAssemblyEnd(a));
  PetscCall(VecView(a,PETSC_VIEWER_STDOUT_WORLD));

  /* assemble b */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "b = [e f] \n"));
  tmp_buf[0] = e; tmp_buf[1] = f;

  PetscCall(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&b));
  PetscCall(VecView(b,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "b = [f e] \n"));
  PetscCall(VecNestSetSubVec(b, 1, e));
  PetscCall(VecNestSetSubVec(b, 0, f));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(VecView(b,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X = [a b] \n"));
  tmp_buf[0] = a; tmp_buf[1] = b;

  PetscCall(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&X));
  PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDot(X,X, &val));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X.X = %g \n", (double)PetscRealPart(val)));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X = [b a] \n"));
  /* re-order components of X */
  PetscCall(VecNestSetSubVec(X,1,a));
  PetscCall(VecNestSetSubVec(X,0,b));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDot(X,X,&val));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X.X = %g \n", (double)PetscRealPart(val)));

  /* re-assemble X */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X = [g h] \n"));
  PetscCall(VecNestSetSubVec(X,1,g));
  PetscCall(VecNestSetSubVec(X,0,h));
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDot(X,X,&val));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "X.X = %g \n", (double)PetscRealPart(val)));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Y = X \n"));
  PetscCall(VecDuplicate(X, &Y));
  PetscCall(VecCopy(X,Y));
  PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDot(Y,Y,&val));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Y.Y = %g \n", (double)PetscRealPart(val)));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Y = [a b] \n"));
  tmp_buf[0] = a; tmp_buf[1] = b;
  tmp_ind[0] = 0; tmp_ind[1] = 1;

  PetscCall(VecNestSetSubVecs(Y,2,tmp_ind,tmp_buf));
  PetscCall(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&c));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&e));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&h));
  PetscCall(VecDestroy(&a));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args,(char*)0, help));
  PetscCall(test_vec_ops());
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
