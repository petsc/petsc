
static char help[] = "Model multi-physics solver\n\n";

/*
     A model "multi-physics" solver based on the Vincent Mousseau's reactor core pilot code.

     There a three grids:

            ----------------------     ---------------
       |   |                     /    /              /
       |   |                     /    /              /
       |   |                     /    /              /
       |   |                     /    /              /
       |   |                     /    /              /
            ----------------------     ---------------

   A 1d grid along the left edge, a 2d grid in the middle and another 2d grid on the right.

*/

#include "petscdmmg.h"


extern PetscErrorCode FormInitialGuessLocal(DALocalInfo*,PetscScalar*,DALocalInfo*,PetscScalar**,DALocalInfo*,PetscScalar**);
extern PetscErrorCode FormFunctionLocal(DALocalInfo*,PetscScalar*,DALocalInfo*,PetscScalar**,DALocalInfo*,PetscScalar**);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;               /* multilevel grid structure */
  PetscErrorCode ierr;
  MPI_Comm       comm;
  DA             da;
  VecPack        pack;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;


  PreLoadBegin(PETSC_TRUE,"SetUp");

    /*
       Create the VecPack object to manage the three grids/physics. 
       We only support a 1d decomposition along the y direction (since one of the grids is 1d).

    */
    ierr = VecPackCreate(comm,&pack);CHKERRQ(ierr);
    ierr = DACreate1d(comm,DA_NONPERIODIC,-6,1,1,0,&da);CHKERRQ(ierr);
    ierr = VecPackAddDA(pack,da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-6,-6,1,PETSC_DETERMINE,1,1,0,0,&da);CHKERRQ(ierr);
    ierr = VecPackAddDA(pack,da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-6,-6,1,PETSC_DETERMINE,1,1,0,0,&da);CHKERRQ(ierr);
    ierr = VecPackAddDA(pack,da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
   
    /*
       Create the solver object and attach the grid/physics info 
    */
    ierr = DMMGCreate(comm,2,0,&dmmg);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)pack);CHKERRQ(ierr);
    ierr = VecPackDestroy(pack);CHKERRQ(ierr);
    CHKMEMQ;


    ierr = DMMGSetInitialGuessLocal(dmmg,(PetscErrorCode (*)(void)) FormInitialGuessLocal);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = DMMGSetSNESLocal(dmmg,(PetscErrorCode (*)(void))0,0,0,0);CHKERRQ(ierr);
    CHKMEMQ;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PreLoadStage("Solve");
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  PreLoadEnd();

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */


#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocal"
/* 
   FormInitialGuessLocal - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuessLocal(DALocalInfo *info1,PetscScalar *x1,DALocalInfo *info2,PetscScalar **x2,DALocalInfo *info3,PetscScalar **x3)
{
  PetscInt       i;
  PetscReal      hx,dhx;

  PetscFunctionBegin;
  dhx = (PetscReal)(info1->mx-1);
  hx = 1.0/dhx;                 

  for (i=info1->xs; i<info1->xs+info1->xm; i++) {
    x1[i]   = 22.3;
  }
  PetscFunctionReturn(0);
}
 

