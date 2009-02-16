/*T
   Concepts: BiGraph
   Processors: 1
T*/

/*
  Create a series of BiGraphs and perform basic queries on them.
  Note: may not fail in parallel, but is not designed to run that way.
*/

static char help[] = "Constructs a series of BiGraphs and performs basic queries on them.\n\n";

#include <BiGraph.hh>

typedef ALE::Two::BiGraph<int,int,int> BiGraphInt3;
typedef std::set<int> int_set;

PetscErrorCode   testBiGraphDiv2();
void             viewConesAndSupports(const ALE::Obj<BiGraphInt3>& bg, const char* name);
void             removeArrows(const ALE::Obj<BiGraphInt3>& bg,         const char* name);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscTruth     flag;
  PetscInt       verbosity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  ierr = testBiGraphDiv2();                                              CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* main() */


#undef  __FUNCT__
#define __FUNCT__ "testBiGraphDiv2"
PetscErrorCode testBiGraphDiv2() {
  PetscInt debug;
  PetscTruth flag;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  debug = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-debug", &debug, &flag);CHKERRQ(ierr);
  ALE::Obj<BiGraphInt3> bg = BiGraphInt3(PETSC_COMM_SELF, debug);
  
  // Add arrows from the first 10 integers to the first 20 integers, coloring the arrows for 0 (even target) or 1 (odd target) 
  for(int i = 0; i < 10; i++) {
    bg->addArrow(2*i+0, i, 0);
    bg->addArrow(2*i+1, i, 1);
  }
  
  // View
  bg->view(std::cout, "bigraph/2");

  // View cones and supports
  viewConesAndSupports(bg, "bigraph/2");
  
  // Take and view the cone of the whole base
  ALE::Obj<BiGraphInt3::traits::coneSet> cone = bg->cone(bg->base());
  std::cout << "Total cone of bigraph/2" << std::endl;
  std::cout << "[";
  for(BiGraphInt3::traits::coneSet::iterator i = cone->begin(); i != cone->end(); i++) {
    std::cout << " " << *i;
  }
  std::cout << " ]" << std::endl;

  // Take and view the support of the whole cap
  ALE::Obj<BiGraphInt3::traits::supportSet> supp = bg->support(bg->cap());
  std::cout << "Total support of bigraph/2" << std::endl;
  std::cout << "[";
  for(BiGraphInt3::traits::supportSet::iterator i = supp->begin(); i != supp->end(); i++) {
    std::cout << *i;
  }
  std::cout << "]";


  // Change each arrow color to its negative
  BiGraphInt3::baseSequence base = bg->base();
  for(BiGraphInt3::baseSequence::iterator i = base.begin(); i != base.end(); i++) {
    BiGraphInt3::coneSequence cone = bg->cone(*i);
    for(BiGraphInt3::coneSequence::iterator j = cone.begin(); j != cone.end(); j++) {
      bg->setColor(*j,*i,-(j.color()));
    }
  }
  // View
  bg->view(std::cout, "bigraph/2 after negation");

  // Remove the arrow from 4 to 2 with color 0
  BiGraphInt3::traits::arrow_type a(4,2,0), b(4,3,0);
  bg->removeArrow(a);
  bg->removeArrow(b);
  // View
  ostringstream txt;
  txt << "bigraph/2 after removal of arrows " << a << " and " << b;
  bg->view(std::cout, txt.str().c_str());

  // Remove first cone
  {
    BiGraphInt3::baseSequence base = bg->base();
    bg->clearCone(*(base.begin()));
  }

  // View
  bg->view(std::cout, "bigraph/2 after removal of the first cone");

  // Remove first support
  {
    BiGraphInt3::capSequence cap = bg->cap();
    bg->clearSupport(*(cap.begin()));
  }

  // View
  bg->view(std::cout, "bigraph/2 after removal of the first support");

  // Now restrict the base to the [0,8[ interval
  try {
    {
      ALE::Obj<int_set> base = int_set();
      for(int i = 0; i < 8; i++) {
        base->insert(i);
      }
      bg->restrictBase(base);
    }
  }
  catch(ALE::Exception e) {
    std::cout << "Caught exception: " << e.message() << std::endl;
  }

  // View
  bg->view(std::cout, "bigraph/2 after restricting base to [0,8[");

  // Now exclude [0,5[ from the base
  try {
    {
      ALE::Obj<int_set> ebase = int_set();
      for(int i = 0; i < 5; i++) {
        ebase->insert(i);
      }
      bg->excludeBase(ebase);
    }
  }
  catch(ALE::Exception e) {
    std::cout << "Caught exception: " << e.message() << std::endl;
  }

  // View
  bg->view(std::cout, "bigraph/2 after excluding [0,5[ from base");
  

  PetscFunctionReturn(0);
}/* testBiGraphDiv2() */

#undef  __FUNCT__
#define __FUNCT__ "viewConesAndSupports"
void viewConesAndSupports(const ALE::Obj<BiGraphInt3>& bg, const char* name) {
  
  // View the cones for all base points
  std::cout << name << " cones:" << std::endl;
  BiGraphInt3::traits::baseSequence base = bg->base();
  for(BiGraphInt3::traits::baseSequence::traits::iterator i = base.begin(); i != base.end(); i++) {
    BiGraphInt3::traits::coneSequence cone = bg->cone(*i);
    std::cout << *i << ": ";
    cone.view(std::cout, true); 
  }

  // View the supports for all cap points
  std::cout << name << " supports:" << std::endl;
  BiGraphInt3::traits::capSequence cap = bg->cap();
  for(BiGraphInt3::traits::capSequence::traits::iterator i = cap.begin(); i != cap.end(); i++) {
    BiGraphInt3::traits::supportSequence supp = bg->support(*i);
    std::cout << *i << ": ";
    supp.view(std::cout, true); 
  }
}/* viewConesAndSupports() */
