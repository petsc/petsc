
static char help[] = "Tests imbedding DMComposites inside DMComposites.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>

typedef struct {
  DM load;
  DM L1,L2;
} Load;

PetscErrorCode LoadCreate(PetscInt n1, PetscInt n2, Load *load)
{
  PetscErrorCode ierr;

  ierr = PetscNew(&load);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,n1,1,1,NULL,&load->L1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(load->L1);CHKERRQ(ierr);
  ierr = DMSetUp(load->L1);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_SELF,DM_BOUNDARY_NONE,n1,1,1,NULL,&load->L2);CHKERRQ(ierr);
  ierr = DMSetFromOptions(load->L2);CHKERRQ(ierr);
  ierr = DMSetUp(load->L2);CHKERRQ(ierr);
  ierr = DMCompositeCreate(PETSC_COMM_SELF,&load->load);CHKERRQ(ierr);
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

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  UnitedStates   unitedstates;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscFinalize();
  return ierr;
}

