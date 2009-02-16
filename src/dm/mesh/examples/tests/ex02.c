/*T
   Concepts: PreSieve^Sieve stratification
   Concepts: PreSieve^Sieve point height, depth
   Concepts: PreSieve^Sieve diameter
   Concepts: PreSieve^Viewing a Sieve
   Processors: n
T*/

/*
  Construct serial a doublet Sieve (see paper) by building it (1) bottom-up, and (2) top-down.
  Each time view the resulting Sieve after the construction is complete, and list 
  the points stratum-by-stratum (1) in the order of increasing depth, and 
  (2) in the order of increasing height.

  Note: this test might not fail if run in parallel, but may produce unintelligible output.
  
*/

static char help[] = "Constructs the doublet Sieve bottom-up and top-down, viewing the resulting stratum structure each time.\n\n";

#include "petscda.h"
#include "petscviewer.h"
#include <stdlib.h>
#include <string.h>

#include <ALE.hh>
#include <Sieve.hh>

PetscErrorCode createDoubletSieveBottomUp(MPI_Comm comm, ALE::Sieve **doublet_p);
PetscErrorCode createDoubletSieveTopDown(MPI_Comm comm, ALE::Sieve **doublet_p);
PetscErrorCode viewStrata(ALE::Sieve *sieve, const char *name = NULL);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscTruth     flag;
  PetscInt       verbosity;
  ALE::Sieve     *doublet;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = PetscPrintf(comm, "Creating a doublet Sieve bottom-up\n");CHKERRQ(ierr);
  ierr = createDoubletSieveBottomUp(comm, &doublet);               CHKERRQ(ierr);
  doublet->view("Bottom-up Doublet");                       
  ierr = viewStrata(doublet, "Bottom-up Doublet");                 CHKERRQ(ierr);
  delete doublet;

  ierr = PetscPrintf(comm, "Creating a doublet Sieve top-down\n"); CHKERRQ(ierr);
  ierr = createDoubletSieveTopDown(comm, &doublet);                CHKERRQ(ierr);
  doublet->view("Top-down Doublet");                       
  ierr = viewStrata(doublet, "Top-down Doublet");                  CHKERRQ(ierr);
  delete doublet;

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* main() */

#undef  __FUNCT__
#define __FUNCT__ "createDoubletSieveBottomUp"
PetscErrorCode createDoubletSieveBottomUp(MPI_Comm comm, ALE::Sieve **doublet_p) {
  PetscFunctionBegin;
  ALE::Sieve *doublet = new ALE::Sieve(comm);
  *doublet_p = doublet;
  ALE::Point_set cone;
  ALE::Point     p;
  cone.clear(); cone.insert(ALE::Point(0,7)); cone.insert(ALE::Point(0,9));  p = ALE::Point(0,2); doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,7)); cone.insert(ALE::Point(0,8));  p = ALE::Point(0,3); doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,8)); cone.insert(ALE::Point(0,9));  p = ALE::Point(0,4); doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,8)); cone.insert(ALE::Point(0,10)); p = ALE::Point(0,5); doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,9)); cone.insert(ALE::Point(0,10)); p = ALE::Point(0,6); doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,2)); cone.insert(ALE::Point(0,3)); cone.insert(ALE::Point(0,4)); p = ALE::Point(0,0); 
  doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,4)); cone.insert(ALE::Point(0,5)); cone.insert(ALE::Point(0,6)); p = ALE::Point(0,1); 
  doublet->addCone(cone,p);
  
  PetscFunctionReturn(0);
}/* createDoubletSieveBottomUp() */

#undef  __FUNCT__
#define __FUNCT__ "createDoubletSieveTopDown"
PetscErrorCode createDoubletSieveTopDown(MPI_Comm comm, ALE::Sieve **doublet_p) {
  PetscFunctionBegin;
  ALE::Sieve *doublet = new ALE::Sieve(comm);
  *doublet_p = doublet;
  ALE::Point_set cone;
  ALE::Point     p;
  cone.clear(); cone.insert(ALE::Point(0,2)); cone.insert(ALE::Point(0,3)); cone.insert(ALE::Point(0,4)); p = ALE::Point(0,0); 
  doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,4)); cone.insert(ALE::Point(0,5)); cone.insert(ALE::Point(0,6)); p = ALE::Point(0,1); 
  doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,7)); cone.insert(ALE::Point(0,9));  p = ALE::Point(0,2); doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,7)); cone.insert(ALE::Point(0,8));  p = ALE::Point(0,3); doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,8)); cone.insert(ALE::Point(0,9));  p = ALE::Point(0,4); doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,8)); cone.insert(ALE::Point(0,10)); p = ALE::Point(0,5); doublet->addCone(cone,p);
  cone.clear(); cone.insert(ALE::Point(0,9)); cone.insert(ALE::Point(0,10)); p = ALE::Point(0,6); doublet->addCone(cone,p);
  
  PetscFunctionReturn(0);
}/* createDoubletSieveTopDown() */


#undef  __FUNCT__
#define __FUNCT__ "viewStrata"
PetscErrorCode viewStrata(ALE::Sieve *sieve, const char *name) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ostringstream txt;

  txt << "Viewing strata of";
  if(name != NULL) {
    txt << " " << name;
  }
  txt << " sieve of diameter " << sieve->diameter() << "\n";
  ierr = PetscPrintf(sieve->getComm(), txt.str().c_str());CHKERRQ(ierr);

  for(int d = 0; d <= sieve->diameter(); d++) {
    ALE::Point_set stratum = sieve->depthStratum(d);
    ierr = PetscPrintf(sieve->getComm(), "Depth stratum %d:\n", d);CHKERRQ(ierr);
    stratum.view();
  }
  for(int d = 0; d <= sieve->diameter(); d++) {
    ALE::Point_set stratum = sieve->heightStratum(d);
    ierr = PetscPrintf(sieve->getComm(), "Height stratum %d:\n", d);CHKERRQ(ierr);
    stratum.view();
  }
  
  PetscFunctionReturn(0);
}/* viewStrata() */
