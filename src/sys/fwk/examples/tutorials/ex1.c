
static char help[] = \
"Sets up a simple component framework consisting of \n\
 a computational component \"Electrolyte\" and a vizualization component \"Viz\".\n\
 * \"Electrolyte\" is implemented in Python (electrolyte.py); during the \"init\" \n\
  configuration stage, it sets up a DMDA and a Vec over that DMDA that needs \n\
  to be plotted.\n\
 * \"Viz\" is implemented in Python (viz.py).  It depends on \"Electrolyte\", \n\
  so at each stage it is configured after \"Electrolyte.\"\n\
  During the \"init\" stage, \"Viz\" retrieves (and stores) the DMDA and the Vec\n\
  From \"Electrolute\"; during the \"viewRho\" stage, all but last components of the Vec\n\
  are plotted using mayavi.\n\
\n\
 One of the ideas is to facilitate passing of named PetscObjects across library (and language)\n\
 boundary without the need for specialized glue code.\n\
 Another idea is to be able to invoke components in the order dictated by their dependencies.\n\
\n\n";

/*T
   Concepts: PetscFwk^configuring framework components in the dependency order
   Concepts: Python^invoking vizualization capabilities implemented in Python.
   Processors: 1
T*/

/* 
  Include "petscsys.h" so that we can use PetscFwk 
  automatically includes:
     petscfwk.h    - PetscFwk routines
*/
#include "petscsys.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args) {
  PetscFwk           fwk, viz = PETSC_NULL;
  PetscInt           numIter = 10, i;
  PetscErrorCode     ierr;
  PetscFunctionBegin;
  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscFwkCreate(PETSC_COMM_SELF, &fwk);                           CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)fwk, "ex1");                     CHKERRQ(ierr);
  /**/
  ierr = PetscFwkRegisterComponentURL(fwk, "Electrolyte", "electrolyte.py:Electrolyte"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentURL(fwk, "Viz", "viz.py:Viz");                         CHKERRQ(ierr);
  ierr = PetscFwkGetComponent(fwk, "Viz", &viz, PETSC_NULL);                             CHKERRQ(ierr);
  /**/
  ierr = PetscFwkVisit(fwk, "init");      CHKERRQ(ierr);
  /**/
  for(i = 0; i < numIter; ++i) {
    ierr = PetscFwkCall(viz, "viewRho");  CHKERRQ(ierr);
  }
  ierr = PetscFwkDestroy(viz);            CHKERRQ(ierr);
  ierr = PetscFwkDestroy(fwk);            CHKERRQ(ierr);
  PetscFinalize();
  PetscFunctionReturn(0);
}/* main() */

