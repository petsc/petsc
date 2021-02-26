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
  PetscErrorCode ierr;
  PetscInt       tmp_ind[2];
  Vec            tmp_buf[2];

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "============== %s ==============\n",PETSC_FUNCTION_NAME);CHKERRQ(ierr);

  /* create 4 worker vectors */
  ierr = VecCreate(PETSC_COMM_WORLD, &c);CHKERRQ(ierr);
  ierr = VecSetSizes(c, PETSC_DECIDE, 4);CHKERRQ(ierr);
  ierr = VecSetType(c, VECMPI);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &d);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &e);CHKERRQ(ierr);
  ierr = VecDuplicate(c, &f);CHKERRQ(ierr);

  /* create two more workers of different sizes */
  ierr = VecCreate(PETSC_COMM_WORLD, &g);CHKERRQ(ierr);
  ierr = VecSetSizes(g, PETSC_DECIDE, 6);CHKERRQ(ierr);
  ierr = VecSetType(g, VECMPI);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD, &h);CHKERRQ(ierr);
  ierr = VecSetSizes(h, PETSC_DECIDE, 8);CHKERRQ(ierr);
  ierr = VecSetType(h, VECMPI);CHKERRQ(ierr);

  /* set the 6 vectors to some numbers */
  ierr = VecSet(c, 1.0);CHKERRQ(ierr);
  ierr = VecSet(d, 2.0);CHKERRQ(ierr);
  ierr = VecSet(e, 3.0);CHKERRQ(ierr);
  ierr = VecSet(f, 4.0);CHKERRQ(ierr);
  ierr = VecSet(g, 5.0);CHKERRQ(ierr);
  ierr = VecSet(h, 6.0);CHKERRQ(ierr);

  /* assemble a */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "a = [c d] \n");CHKERRQ(ierr);
  tmp_buf[0] = c; tmp_buf[1] = d;

  ierr = VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&a);CHKERRQ(ierr);
  ierr = VecView(a,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "a = [d c] \n");CHKERRQ(ierr);
  ierr = VecNestSetSubVec(a, 1, c);CHKERRQ(ierr);
  ierr = VecNestSetSubVec(a, 0, d);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(a);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(a);CHKERRQ(ierr);
  ierr = VecView(a,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* assemble b */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "b = [e f] \n");CHKERRQ(ierr);
  tmp_buf[0] = e; tmp_buf[1] = f;

  ierr = VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&b);CHKERRQ(ierr);
  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "b = [f e] \n");CHKERRQ(ierr);
  ierr = VecNestSetSubVec(b, 1, e);CHKERRQ(ierr);
  ierr = VecNestSetSubVec(b, 0, f);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "X = [a b] \n");CHKERRQ(ierr);
  tmp_buf[0] = a; tmp_buf[1] = b;

  ierr = VecCreateNest(PETSC_COMM_WORLD,2,NULL,tmp_buf,&X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDot(X,X, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "X.X = %g \n", (double)PetscRealPart(val));CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "X = [b a] \n");CHKERRQ(ierr);
  /* re-order components of X */
  ierr = VecNestSetSubVec(X,1,a);CHKERRQ(ierr);
  ierr = VecNestSetSubVec(X,0,b);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDot(X,X,&val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "X.X = %g \n", (double)PetscRealPart(val));CHKERRQ(ierr);

  /* re-assemble X */
  ierr = PetscPrintf(PETSC_COMM_WORLD, "X = [g h] \n");CHKERRQ(ierr);
  ierr = VecNestSetSubVec(X,1,g);CHKERRQ(ierr);
  ierr = VecNestSetSubVec(X,0,h);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDot(X,X,&val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "X.X = %g \n", (double)PetscRealPart(val));CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Y = X \n");CHKERRQ(ierr);
  ierr = VecDuplicate(X, &Y);CHKERRQ(ierr);
  ierr = VecCopy(X,Y);CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDot(Y,Y,&val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Y.Y = %g \n", (double)PetscRealPart(val));CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Y = [a b] \n");CHKERRQ(ierr);
  tmp_buf[0] = a; tmp_buf[1] = b;
  tmp_ind[0] = 0; tmp_ind[1] = 1;

  ierr = VecNestSetSubVecs(Y,2,tmp_ind,tmp_buf);CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&c);CHKERRQ(ierr);
  ierr = VecDestroy(&d);CHKERRQ(ierr);
  ierr = VecDestroy(&e);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = VecDestroy(&h);CHKERRQ(ierr);
  ierr = VecDestroy(&a);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args,(char*)0, help);if (ierr) return ierr;
  ierr = test_vec_ops();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

TEST*/
