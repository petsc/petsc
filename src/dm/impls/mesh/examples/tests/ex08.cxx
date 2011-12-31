/*T
   Concepts: ArrowContainer, Predicate
   Processors: 1
T*/

/*
  Tests Predicate-enabled ArrowContainers.
*/

static char help[] = "Constructs and views test Predicate-enabled ArrowContainers.\n\n";

#include <Filter.hh>
#include <Sifter.hh>

PetscErrorCode testArrowFilters();
PetscErrorCode testCone();
typedef ALE::Experimental::UniColorArrowSet                                             ArrowSet;
typedef ALE::FilterDef::DummyPredicateSetClearer<ArrowSet>                              ArrowSetPredicateClearer;
typedef ALE::Experimental::SifterDef::ArrowContainer<ArrowSet,ArrowSetPredicateClearer> ArrowContainer;
//
ArrowContainer::filter_object_type requestFilter(ArrowContainer& ac, ArrowContainer::predicate_type width, const char *label = NULL);
void adjustFilter(ArrowContainer& ac, ArrowContainer::filter_object_type& f, ArrowContainer::predicate_type width, const char *label);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscBool      flag;
  PetscInt       verbosity;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  
  try {
    ierr = testArrowFilters();                           CHKERRQ(ierr);
    ierr = testCone();                                   CHKERRQ(ierr);
  }
  catch(const ALE::FilterDef::FilterError& e) {
    std::cout << "FILTER ERROR: " << e.msg() << std::endl;
  }
  catch(const ALE::Exception& e) {
    std::cout << "ERROR: " << e.msg() << std::endl;
  }
  catch(...) {
    std::cout << "SOME KINDA ERROR" << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}/* main() */


#undef __FUNCT__
#define __FUNCT__ "testCone"
PetscErrorCode testCone()
{
  PetscInt       verbosity;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag);CHKERRQ(ierr);

  std::cout <<  std::endl << ">>>>> testCone:" << std::endl;

  ArrowContainer ac;
  std::cout << "ArrowContainer has these FilterContainer parameters: " << ac << std::endl;
  ac.addArrow(0,0,0);
  ac.addArrow(1,0,2);
  ac.addArrow(2,0,1);
  ac.addArrow(2,1,0);
  ac.addArrow(3,1,0);
  ac.addArrow(4,1,0);
  
  ArrowContainer::ConeSequence cone0 = ac.cone(0);
  cone0.view(std::cout, "ac.cone0(0)");
  ArrowContainer::filter_object_type f1 = ac.newFilter(1);
  for(ArrowContainer::ConeSequence::iterator<> ci = cone0.begin(); ci != cone0.end();) {
    std::cout << ">>> cone0.begin(): " << cone0.begin().arrow() << " <";
    std::cout << (ArrowContainer::predicate_traits::printable_type)(cone0.begin().predicate()) << "> " << std::endl;
    std::cout << ">>> ci: " << ci.arrow() << " <";;
    std::cout << (ArrowContainer::predicate_traits::printable_type)(ci.predicate()) << "> " << std::endl;
    std::cout << ">>> cone0.end(): " << cone0.end().arrow() << " <";
    std::cout << (ArrowContainer::predicate_traits::printable_type)(cone0.end().predicate()) << "> " << std::endl;
    
    bool adv = false;
    if(ci.source() > 0) {
      std::cout << "attempting to mark arrow " << ci.arrow() << " with predicate " << 1+ci.source() << " ... ";
      adv = ci.markOut(1+ci.source());
      if(adv) {
        std::cout << " successful";
      }
      else {
        std::cout << " failed";
      }
      std::cout << std::endl;
    }
    else {
      ++ci;
    }
    std::cout << "<<< cone0.begin(): " << cone0.begin().arrow() << " <";
    std::cout << (ArrowContainer::predicate_traits::printable_type)(cone0.begin().predicate()) << "> " << std::endl;
    std::cout << "<<< ci: " << ci.arrow() << " <";
    std::cout << (ArrowContainer::predicate_traits::printable_type)(ci.predicate()) << "> " << std::endl;
    std::cout << "<<< cone0.end(): " << cone0.end().arrow() << " <";
    std::cout << (ArrowContainer::predicate_traits::printable_type)(cone0.end().predicate()) << "> " << std::endl;
    cone0.view(std::cout, "cone0 at the bottom of the loop");
    std::cout << std::endl;
  }
  cone0.view(std::cout, "ac.cone0");
  ArrowContainer::ConeSequence cone01 = ac.cone(0, f1);
  cone01.view(std::cout, "ac.cone(0,f1)");
  ArrowContainer::filter_object_type f2 = ac.newFilter(2);
  ArrowContainer::ConeSequence cone02 = ac.cone(0, f2);
  cone02.view(std::cout, "ac.cone(0,f2)");
  
  std::cout << "<<<<< testCone:" << std::endl << std::endl;
  
  PetscFunctionReturn(0);
}/* testCone() */


#undef __FUNCT__
#define __FUNCT__ "testArrowFilters"
PetscErrorCode testArrowFilters()
{
  PetscInt       verbosity;
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag);CHKERRQ(ierr);

  std::cout << std::endl << ">>>>> testArrowFilters:" << std::endl;

  ArrowContainer ac;
  
  // Check the predicate type
  std::cout << "Using predicate type with the following traits:" << std::endl;
  std::cout << "min = "  << (ArrowContainer::predicate_traits::printable_type)ArrowContainer::predicate_traits::min;
  std::cout << ", max = " << (ArrowContainer::predicate_traits::printable_type)ArrowContainer::predicate_traits::max;
  std::cout << ", third = " << (ArrowContainer::predicate_traits::printable_type)ArrowContainer::predicate_traits::third;
  std::cout << std::endl << std::endl;

  ArrowContainer::predicate_type width = (ArrowContainer::predicate_traits::third-1)/2;
  std::cout << "Will request positive filters from ArrowContainer ac: " << ac << std::endl;
  {
    ArrowContainer::filter_object_type   pf1 = requestFilter(ac,width, "pf1");
    {ArrowContainer::filter_object_type  pf2 = requestFilter(ac,width, "pf2");}
    ArrowContainer::filter_object_type   pf3 = requestFilter(ac,width, "pf3");
    try {
      ArrowContainer::filter_object_type pf4 = requestFilter(ac,width, "pf4");
    }
    catch(const ALE::FilterDef::FilterError& e) {
      std::cout << "FILTER ERROR (not unexpected ;)): " << e.msg() << std::endl; 
    }
  }
  ArrowContainer::filter_object_type pf5 = requestFilter(ac,width, "pf5");
  std::cout << "End-state of ArrowContainer ac: " << ac << std::endl;
  //
  width = -(ArrowContainer::predicate_traits::third-1)/2;
  std::cout << std::endl << "Will request negative filters from ArrowContainer ac: " << ac << std::endl;
  {
    ArrowContainer::filter_object_type  nf1 = requestFilter(ac,width, "nf1");
    {ArrowContainer::filter_object_type nf2 = requestFilter(ac,width, "nf2");}
    ArrowContainer::filter_object_type  nf3 = requestFilter(ac,width, "nf3");
    try {
      ArrowContainer::filter_object_type nf4 = requestFilter(ac,width, "nf4");
    }
    catch(const ALE::FilterDef::FilterError& e) {
      std::cout << "FILTER ERROR (not unexpected ;)): " << e.msg() << std::endl; 
    }
  }
  ArrowContainer::filter_object_type nf5 = requestFilter(ac,width, "nf5");
  std::cout << "End-state of ArrowContainer ac: " << ac << std::endl;

  std::cout << std::endl;
  width = (ArrowContainer::predicate_traits::third-1)/2;
  adjustFilter(ac, pf5, width/2, "pf5");
  adjustFilter(ac, nf5, -width/2, "nf5");
  
  std::cout << "<<<<< testArrowFilters:" << std::endl << std::endl;

  PetscFunctionReturn(0);
}/* testArrowFilters() */

ArrowContainer::filter_object_type requestFilter(ArrowContainer& ac, ArrowContainer::predicate_type width, const char *label) {
  std::cout << "Requesting filter ";
  if(label != NULL) {
    std::cout << label;
  }
  std::cout << " of width " << (ArrowContainer::predicate_traits::printable_type)width << std::endl;
  std::cout << "from ArrowContainer: " << ac << std::endl;
  ArrowContainer::filter_object_type f = ac.newFilter(width);
  std::cout << "resulting filter";
  if(label != NULL) {
    std::cout << " " << label;
  }
  std::cout << ": " << f << std::endl;
  std::cout << "resulting ArrowContainer state: " << ac << std::endl;
  
  return f;
}


void adjustFilter(ArrowContainer& ac, ArrowContainer::filter_object_type& f, ArrowContainer::predicate_type width, const char *label) {
  // 
  bool contract = (width < 0);
  if(contract) {
    std::cout << "Contracting filter ";
  }
  else {
    std::cout << "Extending filter ";
  }
  if(label != NULL) {
    std::cout << label;
  }
  std::cout << " " << f;
  std::cout << " by width ";
  if(contract) {
    std::cout << -(ArrowContainer::predicate_traits::printable_type)width << std::endl;
  }
  else {
    std::cout << (ArrowContainer::predicate_traits::printable_type)width << std::endl;
  }
  std::cout << "in ArrowContainer: " << ac << std::endl;
  if(contract) {
    f->contract(-width);
  }
  else {
    f->extend(width);
  }
  std::cout << "resulting filter";
  if(label != NULL) {
    std::cout << " " << label;
  }
  std::cout << ": " << f << std::endl;
  std::cout << "resulting ArrowContainer state: " << ac << std::endl;
  
}


