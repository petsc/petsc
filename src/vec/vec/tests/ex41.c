static char help[] = "Nest vector set subvector functionality.\n\n";

/*T
   Concepts: vectors^block operators
   Concepts: vectors^setting values
   Concepts: vectors^local access to
   Processors: n
T*/

#include <petscvec.h>

PetscErrorCode test_vec_ops(void)
{
  Vec            X,Y,a,b;
  Vec            c,d,e,f,g,h;
  PetscScalar    val;
  PetscInt       tmp_ind[2];
  Vec            tmp_buf[2];

  PetscFunctionBegin;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "============== %s ==============\n",PETSC_FUNCTION_NAME));

  /* create 4 worker vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &c));
  CHKERRQ(VecSetSizes(c, PETSC_DECIDE, 4));
  CHKERRQ(VecSetType(c, VECMPI));
  CHKERRQ(VecDuplicate(c, &d));
  CHKERRQ(VecDuplicate(c, &e));
  CHKERRQ(VecDuplicate(c, &f));

  /* create two more workers of different sizes */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &g));
  CHKERRQ(VecSetSizes(g, PETSC_DECIDE, 6));
  CHKERRQ(VecSetType(g, VECMPI));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &h));
  CHKERRQ(VecSetSizes(h, PETSC_DECIDE, 8));
  CHKERRQ(VecSetType(h, VECMPI));

  /* set the 6 vectors to some numbers */
  CHKERRQ(VecSet(c, 1.0));
  CHKERRQ(VecSet(d, 2.0));
  CHKERRQ(VecSet(e, 3.0));
  CHKERRQ(VecSet(f, 4.0));
  CHKERRQ(VecSet(g, 5.0));
  CHKERRQ(VecSet(h, 6.0));

  /* assemble a */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "a = [c d] \n"));
  tmp_buf[0] = c; tmp_buf[1] = d;

  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&a));
  CHKERRQ(VecView(a,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "a = [d c] \n"));
  CHKERRQ(VecNestSetSubVec(a, 1, c));
  CHKERRQ(VecNestSetSubVec(a, 0, d));
  CHKERRQ(VecAssemblyBegin(a));
  CHKERRQ(VecAssemblyEnd(a));
  CHKERRQ(VecView(a,PETSC_VIEWER_STDOUT_WORLD));

  /* assemble b */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "b = [e f] \n"));
  tmp_buf[0] = e; tmp_buf[1] = f;

  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&b));
  CHKERRQ(VecView(b,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "b = [f e] \n"));
  CHKERRQ(VecNestSetSubVec(b, 1, e));
  CHKERRQ(VecNestSetSubVec(b, 0, f));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));
  CHKERRQ(VecView(b,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X = [a b] \n"));
  tmp_buf[0] = a; tmp_buf[1] = b;

  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&X));
  CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDot(X,X, &val));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X.X = %g \n", (double)PetscRealPart(val)));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X = [b a] \n"));
  /* re-order components of X */
  CHKERRQ(VecNestSetSubVec(X,1,a));
  CHKERRQ(VecNestSetSubVec(X,0,b));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));
  CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDot(X,X,&val));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X.X = %g \n", (double)PetscRealPart(val)));

  /* re-assemble X */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X = [g h] \n"));
  CHKERRQ(VecNestSetSubVec(X,1,g));
  CHKERRQ(VecNestSetSubVec(X,0,h));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));
  CHKERRQ(VecView(X,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDot(X,X,&val));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "X.X = %g \n", (double)PetscRealPart(val)));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Y = X \n"));
  CHKERRQ(VecDuplicate(X, &Y));
  CHKERRQ(VecCopy(X,Y));
  CHKERRQ(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDot(Y,Y,&val));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Y.Y = %g \n", (double)PetscRealPart(val)));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Y = [a b] \n"));
  tmp_buf[0] = a; tmp_buf[1] = b;
  tmp_ind[0] = 0; tmp_ind[1] = 1;

  CHKERRQ(VecNestSetSubVecs(Y,2,tmp_ind,tmp_buf));
  CHKERRQ(VecView(Y,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&c));
  CHKERRQ(VecDestroy(&d));
  CHKERRQ(VecDestroy(&e));
  CHKERRQ(VecDestroy(&f));
  CHKERRQ(VecDestroy(&g));
  CHKERRQ(VecDestroy(&h));
  CHKERRQ(VecDestroy(&a));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&Y));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{

  CHKERRQ(PetscInitialize(&argc, &args,(char*)0, help));
  CHKERRQ(test_vec_ops());
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
