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

typedef ALE::Experimental::SifterDef::ArrowContainer<ALE::Experimental::TopFilterUniColorArrowSet>
ArrowContainer;
//
ArrowContainer::filter_object_type requestFilter(ArrowContainer& ac, ArrowContainer::predicate_type width);


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
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag); CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  
  try {
    ierr = testArrowFilters();                            CHKERRQ(ierr);
    ierr = testCone();                                    CHKERRQ(ierr);
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


  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* main() */


#undef __FUNCT__
#define __FUNCT__ "testCone"
PetscErrorCode testCone()
{
  PetscInt       verbosity;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag); CHKERRQ(ierr);

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
      adv = ci.markUp(1+ci.source());
      if(adv) {
        std::cout << " successful";
      }
      else {
        std::cout << " failed";
      }
      std::cout << std::endl;
    }
    if(!adv){ci++;}
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
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  verbosity = 1;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-verbosity", &verbosity, &flag); CHKERRQ(ierr);

  std::cout << std::endl << ">>>>> testArrowFilters:" << std::endl;

  ArrowContainer ac;
  
  // Check the predicate type
  std::cout << "Using predicate type with the following traits:" << std::endl;
  std::cout << "half = "  << (ArrowContainer::predicate_traits::printable_type)ArrowContainer::predicate_traits::half;
  std::cout << ", max = " << (ArrowContainer::predicate_traits::printable_type)ArrowContainer::predicate_traits::max;
  std::cout << std::endl << std::endl;

  ArrowContainer::predicate_type width = ArrowContainer::predicate_traits::half-1;
  std::cout << "Will excersize ArrowContainer ac: " << ac << std::endl;
  {
    ArrowContainer::filter_object_type f1 = requestFilter(ac,width);
    {ArrowContainer::filter_object_type f2 = requestFilter(ac,width);}
    ArrowContainer::filter_object_type f3 = requestFilter(ac,width);
    try {
      ArrowContainer::filter_object_type f4 = requestFilter(ac,width);
    }
    catch(const ALE::FilterDef::FilterError& e) {
      std::cout << "FILTER ERROR (not unexpected ;)): " << e.msg() << std::endl; 
    }
  }
  std::cout << "End-state of ArrowContainer ac: " << ac << std::endl;
  
  std::cout << "<<<<< testArrowFilters:" << std::endl << std::endl;

  PetscFunctionReturn(0);
}/* testArrowFilters() */

ArrowContainer::filter_object_type requestFilter(ArrowContainer& ac, ArrowContainer::predicate_type width) {
  std::cout << "Requesting filter of width " << (ArrowContainer::predicate_traits::printable_type)width << std::endl;
  std::cout << "from ArrowContainer: " << ac << std::endl;
  ArrowContainer::filter_object_type f = ac.newFilter(width);
  std::cout << "resulting filter: " << f << std::endl;
  std::cout << "resulting ArrowContainer state: " << ac << std::endl;
  
  return f;
}


