
static char help[] = \
"Sets up two PetscShell objects -- Field and Viz -- and attaches their backends to them.\n\
 * Field's backend is implemented in Python (field.py); in response to an \"init\" call\n\
  it sets up a DMDA and a Vec over that DMDA that needs to be plotted.\n\
 * Viz's backend is implemented in Python (viz1.py).  It does nothing in response to \"init\"\n\
  and in response to \"viewRho\" it looks for composed PetscObjects \"mesh\" and \"rho\" \n\
  and then plots z-slices of \"rho\" (\"mesh\" is assumed to be a 3D DMDA).\n\
 * The code below makes sure that after Field has been initialized, the DMDA and the Vec\n\
  are extracted out of it and composed with Viz under the respective names \"mesh\" and \n\
  \"rho\", before \"viewRho\" is called on it. \n\
  See ex2 for an automated management of dependencies between PetscShells.\n\
\n\
 The main idea here is to facilitate passing of named PetscObjects across library (and language)\n\
 boundaries without the need for specialized glue code.\n\
\n\n";

/*T
   Concepts: PetscShell^configuring an object to process string queries using a Python backend.
   Concepts: Python^invoking computation and vizualization capabilities implemented in Python.
   Processors: 1
T*/

/* 
  Include "petscsys.h" so that we can use PetscShell
  automatically includes:
     petscshell.h    - PetscShell routines
*/
#include <petscsys.h>


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args) {
  PetscShell         field, viz;
  PetscInt           numIter = 10, i;
  PetscObject        mesh, rho;
  PetscErrorCode     ierr;
  PetscFunctionBegin;
  PetscInitialize(&argc,&args,(char *)0,help);
  /**/
  ierr = PetscShellCreate(PETSC_COMM_SELF, &field);       CHKERRQ(ierr);
  ierr = PetscShellSetURL(field, "./field.py:Field");     CHKERRQ(ierr);
  ierr = PetscShellCreate(PETSC_COMM_SELF, &viz);         CHKERRQ(ierr);
  ierr = PetscShellSetURL(viz, "./viz1.py:Viz");          CHKERRQ(ierr);
  /**/
  ierr = PetscShellCall(field, "init");                        CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)field, "mesh", &mesh);  CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)field, "rho",  &rho);   CHKERRQ(ierr);

  ierr = PetscObjectCompose((PetscObject)viz, "mesh", mesh);   CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)viz, "rho",  rho);    CHKERRQ(ierr);

  for(i = 0; i < numIter; ++i) {
    ierr = PetscShellCall(viz, "viewRho");              CHKERRQ(ierr); 
  }
  /**/
  /* ierr = PetscShellDestroy(&viz);                    CHKERRQ(ierr); */
  ierr = PetscShellDestroy(&field);                  CHKERRQ(ierr);
  PetscFinalize();
  PetscFunctionReturn(0);
}/* main() */

