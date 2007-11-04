#ifndef included_ALE_xsifterTest_hh
#define included_ALE_xsifterTest_hh

#include <XSifter.hh>

namespace ALE {
  namespace Test {
    struct XSifterTester {
      typedef ALE::XSifterDef::Arrow<double,int,char>    default_arrow_type;
      typedef ALE::XSifter<default_arrow_type>           default_xsifter_type;
      //
      ALE::Component::ArgDB argDB;
      XSifterTester() : argDB("XSifter basic test options"){
        argDB("debug", ALE::Component::Arg<int>().DEFAULT(0));
        argDB("codebug", "codebug level", ALE::Component::Arg<int>().DEFAULT(0));
        argDB("capSize", "The size of Fork xsifter cap", ALE::Component::Arg<int>().DEFAULT(3));
        argDB("baseSize", "The size of Fork xsifter base", ALE::Component::Arg<int>().DEFAULT(10));
        argDB("iterations", "The number of test repetitions", ALE::Component::Arg<int>().DEFAULT(1));
        argDB("marker", "The marker to apply to slice members", ALE::Component::Arg<int>().DEFAULT(0));
      };
      //
      #undef __FUNCT__
      #define __FUNCT__ "createForkXSifter"
      static ALE::Obj<default_xsifter_type> createForkXSifter(const MPI_Comm& comm, const ALE::Component::ArgDB& argDB) {
        typedef default_xsifter_type              xsifter_type;
        typedef xsifter_type::arrow_type          arrow_type;
        typedef std::set<arrow_type::target_type> RealBase;
        typedef std::set<arrow_type::source_type> RealCone;
        ALE::Obj<xsifter_type>   xsifter = new default_xsifter_type(comm, argDB["debug"]);
        for(int i = 0; i < (int)argDB["baseSize"]; i++) {
          // Add an arrow from i mod baseSize to i with color 'Y'.
          xsifter->addArrow(arrow_type((double)(i%(int)argDB["capSize"]),i,'Y'));
        }
        return xsifter;
      };// createForkXSifter()
      //
      #undef __FUNCT__
      #define __FUNCT__ "createHatXSifter"
      static ALE::Obj<default_xsifter_type> createHatXSifter(const MPI_Comm& comm, const ALE::Component::ArgDB& argDB) {
        typedef default_xsifter_type                      xsifter_type;
        typedef xsifter_type::arrow_type                  arrow_type;
        typedef std::set<arrow_type::target_type>         RealBase;
        typedef std::set<arrow_type::source_type>         RealCone;
        ALE::Obj<xsifter_type>   xsifter = new default_xsifter_type(comm, argDB["debug"]);
        for(int i = 0; i < (int)argDB["baseSize"]; i++) {
          // Add an arrow from i to i mod baseSize with  color 'H'.
          xsifter->addArrow(arrow_type((double)i,i%(int)argDB["capSize"],'H'));
        }
        return xsifter;
      };// createHatXSifter()
      //
      #undef __FUNCT__
      #define __FUNCT__ "BasicTest"
      template <typename XSifter_>
      static PetscErrorCode BasicTest(const ALE::Obj<XSifter_>& xsifter, const ALE::Component::ArgDB& argDB, const char* xsifterName = NULL)
      {
        typedef XSifter_                                   xsifter_type;
        typedef typename xsifter_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;
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
      template <typename XSifter_>
      static PetscErrorCode BaseRangeTest(const ALE::Obj<XSifter_>& xsifter, const ALE::Component::ArgDB& argDB, const char* xsifterName = NULL)
      {
        typedef XSifter_                                   xsifter_type;
        typedef typename xsifter_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;
        ALE::Obj<typename xsifter_type::BaseSequence> base = xsifter->base(); 
        PetscFunctionBegin;
        ALE::LogStage stage = ALE::LogStageRegister("Base Test");
        ALE::LogStagePush(stage);
        if(xsifterName != NULL) {
          std::cout << xsifterName << ": ";
        }
        typename xsifter_type::BaseSequence::iterator bbegin, bend;
        bbegin = base->begin();
        bend   = base->end();
        std::cout << "Base: *bbegin: " << *bbegin << ", *bend: " << *bend << std::endl;
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }//BaseRangeTest()
      //
      #undef __FUNCT__
      #define __FUNCT__ "BaseTest"
      template <typename XSifter_>
      static PetscErrorCode BaseTest(const ALE::Obj<XSifter_>& xsifter, ALE::Component::ArgDB, const char* xsifterName = NULL)
      {        
        typedef XSifter_                                   xsifter_type;
        typedef typename xsifter_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;
        //
        ALE::Obj<typename xsifter_type::BaseSequence> base;
        base = xsifter->base();
        PetscFunctionBegin;
        ALE::LogStage stage = ALE::LogStageRegister("Base Test");
        ALE::LogStagePush(stage);
        if(xsifterName != NULL) {
          std::cout << xsifterName << ": ";
        }
        RealBase realBase;
        for(typename xsifter_type::iterator ii = xsifter->begin(); ii != xsifter->end(); ++ii){
          realBase.insert(ii->target());
        }
        typename xsifter_type::BaseSequence::iterator bbegin, bend, itor;
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
        for(typename RealBase::iterator ii = realBase.begin(); ii!=realBase.end(); ++ii) {   
          std::cout << *ii << ", ";
        }
        std::cout << std::endl;
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }//BaseTest()
      //
      #undef __FUNCT__
      #define __FUNCT__ "ConeTest"
      template <typename XSifter_>
      static PetscErrorCode ConeTest(const ALE::Obj<XSifter_>& xsifter, const ALE::Component::ArgDB& argDB, const char* xsifterName = NULL)
      {
        typedef XSifter_                                   xsifter_type;
        typedef typename xsifter_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;

        ALE::Obj<typename xsifter_type::BaseSequence> base = xsifter->base();
        
        PetscFunctionBegin;
        
        ALE::LogStage stage = ALE::LogStageRegister("Cone Test");
        ALE::LogStagePush(stage);
        if(xsifterName != NULL) {
          std::cout << xsifterName << ": " << std::endl;
        }
        std::cout << "Base:" << std::endl;
        typename xsifter_type::BaseSequence::iterator b_begin, b_end, b_itor, b_next;
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
          ALE::Obj<typename xsifter_type::ConeSequence> cone;
          cone = xsifter->cone(*b_itor);
          
          // build realCone
          RealCone realCone;
          for(typename xsifter_type::iterator ii = xsifter->begin(); ii != xsifter->end(); ++ii){
            if(ii->target() == *b_itor) {
              realCone.insert(ii->source());
            }
          }  
          // display cone
          typename xsifter_type::ConeSequence::iterator c_begin, c_end, c_itor;
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
          for(typename RealCone::iterator ii = realCone.begin(); ii!=realCone.end(); ++ii) {   
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
      #define __FUNCT__ "SliceBasicTest"
      template <typename XSifter_>
      static PetscErrorCode SliceBasicTest1(const ALE::Obj<XSifter_>& xsifter, const ALE::Component::ArgDB& argDB, const char* xsifterName = NULL)
      {
        typedef XSifter_                                   xsifter_type;
        typedef typename xsifter_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;
        PetscFunctionBegin;
        ALE::LogStage stage = ALE::LogStageRegister("BasicTest1");
        ALE::LogStagePush(stage);
        {// first slice scope
          std::cout << "First slice scope:\n";
          try {
            string sliceName("'Total1' of ");
            if(xsifterName != NULL) {sliceName = sliceName + string(xsifterName) + string(" ");};
            string label;
            std::cout << "Attempting to allocate an ArrowSlice " << sliceName << "\n";
            typename xsifter_type::ArrowSlice slice(xsifter->slice());
            std::cout << "No problem!\n";
            //
            label = sliceName + string(" unpopulated");
            slice.view(std::cout, label.c_str());
            //// populating slice
            // traverse the whole sifter and insert the arrows into the slice
            typename xsifter_type::ArrowSlice::marker_type marker = argDB["marker"];
            for(typename xsifter_type::iterator iter = xsifter->begin(); iter != xsifter->end(); ++iter) {
              slice.add(*iter, marker);
            }
            label = sliceName + string("populated");
            slice.view(std::cout, label.c_str());
            slice.clean();
            label = sliceName + string("cleaned");
            slice.view(std::cout, label.c_str());
            try {
              std::cout << "Attempting to allocate another slice:\n";
              xsifter->slice();
              std::cout << "No problem!\n";
            }
            catch(ALE::XSifterDef::NoSlices e) {
              std::cout << "Caught a 'NoSlices' exception\n";
            }
          } catch(ALE::XSifterDef::NoSlices e) {
            std::cout << "Caught a 'NoSlices' exception\n";
          }
        }// first slice scope
        {
          std::cout << "Second slice scope:\n"; 
          try {
            string sliceName("'Total2' of ");
            if(xsifterName != NULL) {sliceName = sliceName + string(xsifterName) + string(" ");};
            string label;
            std::cout << "Attempting to allocate an ArrowSlice " << sliceName << "\n";
            typename xsifter_type::ArrowSlice slice(xsifter->slice());
            std::cout << "No problem!\n";
            label = sliceName + string(" unpopulated");
            slice.view(std::cout, label.c_str());
            try {
              std::cout << "Attempting to allocate another slice:\n";
              xsifter->slice();
              std::cout << "No problem!\n";
            }
            catch(ALE::XSifterDef::NoSlices e) {
              std::cout << "Caught a 'NoSlices' exception\n";
            }
          }
          catch(ALE::XSifterDef::NoSlices e) {
            std::cout << "Caught a 'NoSlices' exception\n";
          }
        }// second slice scope
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }// SliceBasicTest1()

    };// struct XSifterTester
  };//namespace Test
};// namespace ALE

#endif
