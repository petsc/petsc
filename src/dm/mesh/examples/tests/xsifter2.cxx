static char help[] = "Sifter Cone and Support Tests.\n\n";

#include <petsc.h>
#include "xsifterTest.hh"

typedef ALE::Test::XSifter::arrow_type   arrow_type;
typedef ALE::Test::XSifter::xsifter_type xsifter_type;
typedef ALE::Test::XSifter::RealCone     RealCone;

#undef __FUNCT__
#define __FUNCT__ "ConeTest"
#undef  __ALE_XDEBUG__ 
#define __ALE_XDEBUG__ 4
PetscErrorCode ConeTest(const ALE::Obj<xsifter_type>& xsifter, ALE::Test::XSifter::Options options, const char* xsifterName = NULL)
{
  ALE::Obj<xsifter_type::BaseSequence> base = xsifter->base();

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("Cone Test");
  ALE::LogStagePush(stage);
  if(xsifterName != NULL) {
    std::cout << xsifterName << ": " << std::endl;
  }
  std::cout << "Base:" << std::endl;
  xsifter_type::BaseSequence::iterator b_begin, b_end, b_itor, b_next;
  b_begin = base->begin();
  b_end   = base->end();
  if(ALE_XDEBUG(__ALE_XDEBUG__)) {
    std::cout << __FUNCT__ ": *b_begin = " << *b_begin << std::endl;
    std::cout << __FUNCT__ ": *b_end   = " << *b_end << std::endl;
  }
  b_itor = b_begin;
  for(; b_itor != b_end; ++b_itor) {   
    std::cout << *b_itor << ", ";
  }
  //
  std::cout << std::endl;
  b_itor  = b_begin;
  for(; b_itor != b_end; ++b_itor) {// base loop
    b_next = b_itor; ++b_next;
    if(ALE_XDEBUG(__ALE_XDEBUG__)) {
      std::cout << __FUNCT__ ": base loop iteration" << std::endl;
      std::cout << __FUNCT__ ": initial *b_begin  = " << *b_begin << std::endl;
      std::cout << __FUNCT__ ": initial *b_end    = " << *b_end << std::endl;
      std::cout << __FUNCT__ ": initial *b_itor   = " << *b_itor << std::endl;
      std::cout << __FUNCT__ ": initial *b_next   = " << *b_next << std::endl;
    }
    // build cone
    ALE::Obj<xsifter_type::ConeSequence> cone;
    if(options.predicate < 0) {
      cone = xsifter->cone(*b_itor);;
    }
    else {
      cone = xsifter->cone(*b_itor, (xsifter_type::predicate_type) options.predicate);
    }
    // build realCone
    RealCone realCone;
    for(xsifter_type::iterator ii = xsifter->begin(); ii != xsifter->end(); ++ii){
      if(ii->target() == *b_itor) {
        if(options.predicate < 0 || (xsifter_type::predicate_type) options.predicate == ii->predicate()) {
          realCone.insert(ii->source());
        }
      }
    }  
    // display cone
    xsifter_type::ConeSequence::iterator c_begin, c_end, c_itor;
    c_begin = cone->begin();
    c_end   = cone->end();
    c_itor   = c_begin;
    std::cout << "    cone(" << *b_itor << ")";
    if(options.predicate >= 0) {
      std::cout << " (predicate: " << options.predicate << ")";
    }
    std::cout << ": [";
    for(; c_itor != c_end; ) {
      if(ALE_XDEBUG(__ALE_XDEBUG__)) {
        std::cout << __FUNCT__ ": cone loop iteration >>>" << std::endl;
        std::cout << __FUNCT__ ": initial *c_begin  = " << *c_begin << std::endl;
        std::cout << __FUNCT__ ": initial *c_itor   = " << *c_itor   << std::endl;
        std::cout << __FUNCT__ ": initial *c_end    = " << *c_end   << std::endl;
      }
      std::cout << *c_itor << ", ";
      ++c_itor;
      if(ALE_XDEBUG(__ALE_XDEBUG__)) {        
        b_next = b_itor; ++b_next;
        std::cout << std::endl;
        std::cout << __FUNCT__ ": *b_begin  = " << *b_begin << std::endl;
        std::cout << __FUNCT__ ": *b_end    = " << *b_end << std::endl;
        std::cout << __FUNCT__ ": *b_itor   = " << *b_itor << std::endl;
        std::cout << __FUNCT__ ": *b_next   = " << *b_next << std::endl; 
        //
        std::cout << __FUNCT__ ": initial *c_begin  = " << *c_begin << std::endl;
        std::cout << __FUNCT__ ": initial *c_itor   = " << *c_itor   << std::endl;
        std::cout << __FUNCT__ ": initial *c_end    = " << *c_end   << std::endl;
        std::cout << __FUNCT__ ": cone loop iteration <<<" << std::endl;
      }
    }
    std::cout << "]" << std::endl;
    if(ALE_XDEBUG(__ALE_XDEBUG__)) {
      std::cout << __FUNCT__ ": final *b_begin  = " << *b_begin << std::endl;
      std::cout << __FUNCT__ ": final *b_end    = " << *b_end << std::endl;
      std::cout << __FUNCT__ ": final *b_itor   = " << *b_itor << std::endl;
      std::cout << __FUNCT__ ": base iteration <<<" << std::endl;
    }
    // display realCone
    std::cout << "realCone(" << *b_itor << ")";
    if(options.predicate >= 0) {
      std::cout << " (predicate: " << options.predicate << ")";
    }
    std::cout << ": [";
    for(RealCone::iterator ii = realCone.begin(); ii!=realCone.end(); ++ii) {   
      std::cout << *ii << ", ";
    }
    std::cout << "] " << std::endl;
  }// base loop
  std::cout << std::endl;
  ALE::LogStagePop(stage);
  PetscFunctionReturn(0);
}// ConeTest()


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help); CHKERRQ(ierr);
  {
    ALE::Test::XSifter::Options        options;
    ALE::Obj<xsifter_type> xsifterFork = ALE::Test::XSifter::createForkXSifter(PETSC_COMM_SELF, options);
    ierr = ConeTest(xsifterFork, options, "Fork XSifter"); CHKERRQ(ierr);
    ALE::Obj<xsifter_type> xsifterHat = ALE::Test::XSifter::createHatXSifter(PETSC_COMM_SELF, options);
    ierr = ConeTest(xsifterHat, options, "Hat XSifter"); CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
