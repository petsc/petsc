#ifndef included_ALE_xsifterTest_hh
#define included_ALE_xsifterTest_hh

#include <petsc.h>
#include <XSifter.hh>

namespace ALE {
  namespace Test {
    struct XSifterTester {
      typedef ALE::XSifterDef::Arrow<double,int,char>    arrow_type;
      typedef ALE::XSifter<arrow_type>                   xsifter_type;
      typedef std::set<arrow_type::target_type>          RealBase;
      typedef std::set<arrow_type::source_type>          RealCone;
      //
      ALE::Component::ArgDB argDB;
      XSifterTester() : argDB("XSifter basic test options"){
        argDB("debug", ALE::Component::Arg<int>().DEFAULT(0));
        argDB("codebug", "codebug level", ALE::Component::Arg<int>().DEFAULT(0));
        argDB("capSize", "The size of Fork xsifter cap", ALE::Component::Arg<int>().DEFAULT(3));
        argDB("baseSize", "The size of Fork xsifter base", ALE::Component::Arg<int>().DEFAULT(10));
        argDB("iterations", "The number of test repetitions", ALE::Component::Arg<int>().DEFAULT(1));
      };
      //
      static ALE::Obj<xsifter_type> createForkXSifter(const MPI_Comm& comm, const ALE::Component::ArgDB& argDB) {
        ALE::Obj<xsifter_type>   xsifter = new xsifter_type(comm, argDB["debug"]);
        for(int i = 0; i < (int)argDB["baseSize"]; i++) {
          // Add an arrow from i mod baseSize to i with color 'Y'.
          xsifter->addArrow(arrow_type((double)(i%(int)argDB["capSize"]),i,'Y'));
        }
        return xsifter;
      };// createForkXSifter()
      //
      static ALE::Obj<xsifter_type> createHatXSifter(const MPI_Comm& comm, const ALE::Component::ArgDB& argDB) {
        ALE::Obj<xsifter_type>   xsifter = new xsifter_type(comm, argDB["debug"]);
        for(int i = 0; i < (int)argDB["baseSize"]; i++) {
          // Add an arrow from i to i mod baseSize with  color 'H'.
          xsifter->addArrow(arrow_type((double)i,i%(int)argDB["capSize"],'H'));
        }
        return xsifter;
      };// createHatXSifter()
      //
      #undef __FUNCT__
      #define __FUNCT__ "BasicTest"
      static PetscErrorCode BasicTest(const ALE::Obj<xsifter_type>& xsifter, const ALE::Component::ArgDB& argDB, const char* xsifterName = NULL)
      {
        PetscFunctionBegin;
        ALE::LogStage stage = ALE::LogStageRegister("Basic Test");
        ALE::LogStagePush(stage);
        xsifter->view(std::cout, xsifterName);
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }// BasicTest()
      //
      #undef __FUNCT__
      #define __FUNCT__ "BaseRangeTest"
      static PetscErrorCode BaseRangeTest(const ALE::Obj<xsifter_type>& xsifter, const ALE::Component::ArgDB& argDB, const char* xsifterName = NULL)
      {
        ALE::Obj<xsifter_type::BaseSequence> base = xsifter->base(); 
        PetscFunctionBegin;
        ALE::LogStage stage = ALE::LogStageRegister("Base Test");
        ALE::LogStagePush(stage);
        if(xsifterName != NULL) {
          std::cout << xsifterName << ": ";
        }
        xsifter_type::BaseSequence::iterator bbegin, bend;
        bbegin = base->begin();
        bend   = base->end();
        std::cout << "Base: *bbegin: " << *bbegin << ", *bend: " << *bend << std::endl;
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }//BaseRangeTest()
      //
      #undef __FUNCT__
      #define __FUNCT__ "BaseTest"
      static PetscErrorCode BaseTest(const ALE::Obj<xsifter_type>& xsifter, ALE::Component::ArgDB, const char* xsifterName = NULL)
      {        
        ALE::Obj<xsifter_type::BaseSequence> base;
        base = xsifter->base();
        PetscFunctionBegin;
        ALE::LogStage stage = ALE::LogStageRegister("Base Test");
        ALE::LogStagePush(stage);
        if(xsifterName != NULL) {
          std::cout << xsifterName << ": ";
        }
        RealBase realBase;
        for(xsifter_type::iterator ii = xsifter->begin(); ii != xsifter->end(); ++ii){
          realBase.insert(ii->target());
        }
        xsifter_type::BaseSequence::iterator bbegin, bend, itor;
        bbegin = base->begin();
        bend   = base->end();
        itor   = bbegin;
        std::cout << "Base: " << std::endl;
        
        int counter = 0;
        for(; itor != bend; ++itor) {   
          std::cout << *itor << ", ";
          counter++;
        }
        std::cout << std::endl;
        std::cout << "realBase: " << std::endl;
        for(RealBase::iterator ii = realBase.begin(); ii!=realBase.end(); ++ii) {   
          std::cout << *ii << ", ";
        }
        std::cout << std::endl;
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }//BaseTest()
      //
      static PetscErrorCode ConeTest(const ALE::Obj<xsifter_type>& xsifter, const ALE::Component::ArgDB& argDB, const char* xsifterName = NULL)
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
        b_itor = b_begin;
        for(; b_itor != b_end; ++b_itor) {   
          std::cout << *b_itor << ", ";
        }
        //
        std::cout << std::endl;
        b_itor  = b_begin;
        for(; b_itor != b_end; ++b_itor) {// base loop
          b_next = b_itor; ++b_next;
          // build cone
          ALE::Obj<xsifter_type::ConeSequence> cone;
          cone = xsifter->cone(*b_itor);
          
          // build realCone
          RealCone realCone;
          for(xsifter_type::iterator ii = xsifter->begin(); ii != xsifter->end(); ++ii){
            if(ii->target() == *b_itor) {
              realCone.insert(ii->source());
            }
          }  
          // display cone
          xsifter_type::ConeSequence::iterator c_begin, c_end, c_itor;
          c_begin = cone->begin();
          c_end   = cone->end();
          c_itor   = c_begin;
          std::cout << "    cone    (" << *b_itor << ")";
          std::cout << ": [";
          for(; c_itor != c_end; ) {
            std::cout << *c_itor << ", ";
            ++c_itor;
          }
          std::cout << "]" << std::endl;
          // display realCone
          std::cout << "    realCone(" << *b_itor << ")";
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
      //
      #undef __FUNCT__
      #define __FUNCT__ "ClosureTest"
      static PetscErrorCode ClosureTest(const ALE::Obj<xsifter_type>& xsifter, ALE::Component::ArgDB, const char* xsifterName = NULL)
      {        
        ALE::Obj<xsifter_type::BaseSequence> base;
        base = xsifter->base();
        PetscFunctionBegin;
        ALE::LogStage stage = ALE::LogStageRegister("Closure Test");
        ALE::LogStagePush(stage);
        if(xsifterName != NULL) {
          std::cout << xsifterName << ": ";
        }
        xsifter_type::BaseSequence::iterator bbegin, bend, itor;
        bbegin = base->begin();
        bend   = base->end();
        itor   = bbegin;
        for(; itor != bend; ++itor) {   
          xsifter_type::target_type t = *itor;
          // Calculate closure here
          std::cout << "cl(" << t << ")= " << "[ ";
          std::cout << "]\n";
        }
        std::cout << "\n";
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }// ClosureTest()
    };// struct XSifterTester
  };//namespace Test
};// namespace ALE

#endif
