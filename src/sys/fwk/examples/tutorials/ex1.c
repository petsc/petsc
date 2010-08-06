#define PETSC_DLL
static char help[] = \
"Sets up a simple component framework consisting of \n\
 a computational component \"Electrolyte\" and a vizualization component \"Viz\".\n\
 * \"Electrolyte\" is implemented in C (this file); during the \"init\" \n\
  configuration stage, it sets up a DA and a Vec over that DA that needs \n\
  to be plotted; nothing is done at configuration stages.\n\
 * \"Viz\" is implemented in Python in viz.py.  It depends on \"Electrolyte\", \n\
  so at each stage it is configured after \"Electrolyte.\"\n\
  During the \"init\" stage, \"Viz\" retrieves (and stores) the DA and the Vec\n\
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
     petscda.h     - DA, Vec
     math.h        - sin, PI
*/
#include "petscsys.h"
#include "petscda.h"
  
#define PI M_PI

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args) {
  PetscFwk           fwk;
  PetscInt           numIter = 10, i;
  PetscErrorCode     ierr;
  PetscFunctionBegin;
  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscFwkCreate(PETSC_COMM_SELF, &fwk);                        CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, "Electrolyte", "Electrolyte"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, "Viz", "viz.py:Viz");       CHKERRQ(ierr);
  ierr = PetscFwkConfigure(fwk, "init");  CHKERRQ(ierr);
  for(i = 0; i < numIter; ++i) {
    ierr = PetscFwkConfigure(fwk, "viewRho"); CHKERRQ(ierr);
  }
  ierr = PetscFwkDestroy(fwk);            CHKERRQ(ierr);
  PetscFinalize();
  PetscFunctionReturn(0);
}/* main() */

#undef __FUNCT__
#define __FUNCT__ "PetscFwkConfigureElectrolyte"
PetscErrorCode PETSC_DLLEXPORT PetscFwkConfigureElectrolyte(PetscFwk fwk, const char* key, const char* stage, PetscObject *_component) {
  DA da;
  Vec rhoVec;
  PetscFwk e;
  PetscInt n = 32, d = 2, Nx = n, Ny = n, Nz = n;
  PetscInt i,j,k,s;
  PetscScalar *rho;
  PetscTruth init;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(*_component == PETSC_NULL) {
    ierr = PetscFwkCreate(((PetscObject)fwk)->comm, &e); CHKERRQ(ierr);
    *_component = (PetscObject)e;
  }
  else {
    /* FIX: should verify the type of *_component? */
    e = (PetscFwk)*_component;
  }
  ierr = PetscStrcmp(stage, "init", &init); CHKERRQ(ierr);
  if(init) {
    ierr = DACreate3d(((PetscObject)e)->comm, DA_NONPERIODIC, DA_STENCIL_BOX, 
                    Nx, Ny, Nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 
                    d+1, 0, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da); 
    CHKERRQ(ierr);
    ierr = VecCreateSeq(((PetscObject)e)->comm,Nx*Ny*Nz*d, &rhoVec); CHKERRQ(ierr);
    ierr = VecGetArray(rhoVec,&rho);          CHKERRQ(ierr);
    for(i = 0; i < Nx; ++i) {
      for(j = 0; j < Ny; ++j) {
        for(k = 0; k < Nz; ++k) {
          for(s = 0; s < d; ++s) {
            rho[((k*Ny+j)*Nx+i)*d+s] = sin(2*PI*i/Nx)*sin(2*PI*j/Ny)*sin(2*PI*k/Nz);
          }
        }
      }
    }
    ierr = VecRestoreArray(rhoVec,&rho); CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)e,"mesh", (PetscObject)da); CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)e,"rho",  (PetscObject)rhoVec); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkConfigureElectrolyte() */  
