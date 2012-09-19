static char help[] = "Nest vector set subvector functionality.\n\n";

/*T
   Concepts: vectors^block operators
   Concepts: vectors^setting values
   Concepts: vectors^local access to
   Processors: n
T*/

#include <stdio.h>
#include <stdlib.h>

#include <petsc.h>
#include <petscvec.h>

#include "../src/vec/vec/impls/nest/vecnestimpl.h"

#undef __FUNCT__
#define __FUNCT__ "test_vec_ops"
PetscErrorCode test_vec_ops( void )
{
  Vec X,Y,a,b;
  Vec c,d,e,f,g,h;
  PetscScalar val;
  PetscErrorCode ierr;
  PetscInt tmp_ind[2];
  Vec tmp_buf[2];

  PetscFunctionBegin;
  PetscPrintf( PETSC_COMM_WORLD, "============== %s ==============\n",PETSC_FUNCTION_NAME);

  /* create 4 slave vectors */
  ierr = VecCreate( PETSC_COMM_WORLD, &c );CHKERRQ(ierr);
  ierr = VecSetSizes( c, PETSC_DECIDE, 4 );CHKERRQ(ierr);
  ierr = VecSetType( c, VECMPI );CHKERRQ(ierr);
  ierr = VecDuplicate( c, &d );CHKERRQ(ierr);
  ierr = VecDuplicate( c, &e );CHKERRQ(ierr);
  ierr = VecDuplicate( c, &f );CHKERRQ(ierr);

  /* create two more slaves of different sizes */
  ierr = VecCreate( PETSC_COMM_WORLD, &g );CHKERRQ(ierr);
  ierr = VecSetSizes( g, PETSC_DECIDE, 6 );CHKERRQ(ierr);
  ierr = VecSetType( g, VECMPI );CHKERRQ(ierr);
  ierr = VecCreate( PETSC_COMM_WORLD, &h );CHKERRQ(ierr);
  ierr = VecSetSizes( h, PETSC_DECIDE, 8 );CHKERRQ(ierr);
  ierr = VecSetType( h, VECMPI );CHKERRQ(ierr);

  /* set the 6 vectors to some numbers */
  ierr = VecSet( c, 1.0 );CHKERRQ(ierr);
  ierr = VecSet( d, 2.0 );CHKERRQ(ierr);
  ierr = VecSet( e, 3.0 );CHKERRQ(ierr);
  ierr = VecSet( f, 4.0 );CHKERRQ(ierr);
  ierr = VecSet( g, 5.0 );CHKERRQ(ierr);
  ierr = VecSet( h, 6.0 );CHKERRQ(ierr);

  /* assemble a */
  PetscPrintf( PETSC_COMM_WORLD, "a = [c d] \n");
  tmp_buf[0] = c ; tmp_buf[1] = d ;
  ierr = VecCreateNest(PETSC_COMM_WORLD,2,PETSC_NULL,tmp_buf,&a);CHKERRQ(ierr);
  ierr = VecView(a,PETSC_VIEWER_STDOUT_WORLD);
  PetscPrintf( PETSC_COMM_WORLD, "a = [d c] \n");
  ierr = VecNestSetSubVec( a, 1, c );CHKERRQ(ierr);
  ierr = VecNestSetSubVec( a, 0, d );CHKERRQ(ierr);
  ierr = VecAssemblyBegin(a);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(a);CHKERRQ(ierr);
  ierr = VecView(a,PETSC_VIEWER_STDOUT_WORLD);

  /* assemble b */
  PetscPrintf( PETSC_COMM_WORLD, "b = [e f] \n");
  tmp_buf[0] = e ; tmp_buf[1] = f ;
  ierr = VecCreateNest(PETSC_COMM_WORLD,2,PETSC_NULL,tmp_buf,&b);CHKERRQ(ierr);
  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);
  PetscPrintf( PETSC_COMM_WORLD, "b = [f e] \n");
  ierr = VecNestSetSubVec( b, 1, e );CHKERRQ(ierr);
  ierr = VecNestSetSubVec( b, 0, f );CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);

  PetscPrintf( PETSC_COMM_WORLD, "X = [a b] \n");
  tmp_buf[0] = a ; tmp_buf[1] = b ;
  ierr = VecCreateNest(PETSC_COMM_WORLD,2,PETSC_NULL,tmp_buf,&X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecDot( X,X, &val );CHKERRQ(ierr);
  PetscPrintf( PETSC_COMM_WORLD, "X.X = %f \n", val );

  /* re-order components of X */
  PetscPrintf( PETSC_COMM_WORLD, "X = [b a] \n");
  ierr = VecNestSetSubVec(X,1,a);CHKERRQ(ierr);
  ierr = VecNestSetSubVec(X,0,b);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecDot(X,X,&val);CHKERRQ(ierr);
  PetscPrintf( PETSC_COMM_WORLD, "X.X = %f \n", val );

  /* re-assemble X */
  PetscPrintf( PETSC_COMM_WORLD, "X = [g h] \n");
  ierr = VecNestSetSubVec(X,1,g);CHKERRQ(ierr);
  ierr = VecNestSetSubVec(X,0,h);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecDot( X,X,&val );CHKERRQ(ierr);
  PetscPrintf( PETSC_COMM_WORLD, "X.X = %f \n", val );

  PetscPrintf( PETSC_COMM_WORLD, "Y = X \n");
  ierr = VecDuplicate(X, &Y);CHKERRQ(ierr);
  ierr = VecCopy(X,Y);CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecDot( Y,Y,&val );CHKERRQ(ierr);
  PetscPrintf( PETSC_COMM_WORLD, "Y.Y = %f \n", val );

  PetscPrintf( PETSC_COMM_WORLD, "Y = [a b] \n");
  tmp_buf[0] = a; tmp_buf[1] = b ;
  tmp_ind[0] = 0; tmp_ind[1] = 1;
  ierr = VecNestSetSubVecs( Y,2,tmp_ind,tmp_buf );CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);

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

int main( int argc, char **args )
{
  PetscInitialize( &argc, &args,(char *)0, help );

  test_vec_ops();

  PetscFinalize();
  return 0;
}
