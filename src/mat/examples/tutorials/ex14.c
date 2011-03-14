
static char help[] = "Substructuring test of MatFwkAIJ: \
a block matrix with an AIJ-like datastructure keeping track of nonzero blocks.\
Each block is a matrix of (generally) any type.\n\n";

/* 
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers               
*/
#include <petscmat.h>


#undef __FUNCT__
#define __FUNCT__ "TestMatIM"
PetscErrorCode TestMatIM() {
  /*
    o------o-------o-------o------o
  v0|    v3|    v6|||    v9|    v9|
    |  e0  |  e2  |||  e4  |  e4  |
    |      |      |||      |      |
    o------o-------o-------o------o
  v1|    v4|    v7|||   v10|   v10|
    |  e1  |  e3  |||  e5  |  e5  |
    |      |      |||      |      |
     ------ ------- ------- ------
    o------o-------o-------o------o
  v2 ----v5 ----v6- -----v7 ----v8
    |      |      |||      |      |
    |  e0  |  e2  |||  e4  |  e4  |
    |      |      |||      |      |
    o------o-------o-------o------o
  v1|    v4|    v7|||   v10|   v10|
    |  e1  |  e3  |||  e5  |  e5  |
    |      |      |||      |      |
    o------o-------o-------o------o

   */
  Mat M; /* Index Map (IM) matrix */
  PetscFunctionBegin;



  PetscFunctionReturn(0);

}/* TestMatIM() */


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args) {
  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = TestMatFwk1();

  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}


