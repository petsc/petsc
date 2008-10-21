
static char help[] = "Tests imbedding DMComposites inside DMComposites.\n\n";

#include "petscda.h"

typedef struct {
  DM load;
  DM L1,L2;
} Load;

PetscErrorCode LoadCreate(PetscInt n1, PetscInt n2, Load *load)
{
  PetscErrorCode ierr;

  ierr = PetscNew(Load,&load);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_SELF,DA_NONPERIODIC,n1,1,1,PETSC_NULL,(DA*)&load->L1);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_SELF,DA_NONPERIODIC,n1,1,1,PETSC_NULL,(DA*)&load->L2);CHKERRQ(ierr);
  ierr = DMCompositeCreate(PETSC_COMM_SELF,(DMComposite*)&load->load);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  DM network;
  DM n1,n2;
} Network;

typedef struct {
  DM generator;
  DM g1,g2;
} Generator;

typedef struct {
  DM        city;
  Load      load;
  Network   network;
  Generator generator;
} City;

typedef struct {
  DM       state;
  City     *cities;
  PetscInt n;
} State;

typedef struct {
  DM       unitedstates;
  State    *states;
  PetscInt n;
} UnitedStates;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  UnitedStates   unitedstates;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
