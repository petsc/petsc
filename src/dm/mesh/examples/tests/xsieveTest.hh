#ifndef included_ALE_xsieveTest_hh
#define included_ALE_xsieveTest_hh

#include <XSieve.hh>

namespace ALE {
  namespace Test {
    struct XSieveTester {
      typedef ALE::XSieveDef::Arrow<double,int,char>    default_arrow_type;
      typedef ALE::XSieveDef::Arrow<int,int,char>       symmetric_arrow_type;
      typedef ALE::XSieve<default_arrow_type,1>         default_xsieve_type;
      typedef ALE::XSieve<symmetric_arrow_type>         symmetric_xsieve_type;
      //
      ALE::ArgDB argDB;
      XSieveTester() : argDB("XSieve basic test options"){
        argDB("debug", ALE::Arg<int>().DEFAULT(0));
        argDB("codebug", "codebug level", ALE::Arg<int>().DEFAULT(0));
        argDB("capSize", "The size of Fork xsieve cap", ALE::Arg<int>().DEFAULT(3));
        argDB("baseSize", "The size of Fork xsieve base", ALE::Arg<int>().DEFAULT(10));
        argDB("iterations", "The number of test repetitions", ALE::Arg<int>().DEFAULT(1));
        argDB("marker", "The marker to apply to slice members", ALE::Arg<int>().DEFAULT(0));
        argDB("silent", "Whether to generate output during test; useful for timing with iterations > 1", ALE::Arg<bool>().DEFAULT(false));
        argDB("treeDepth", "The depth of the tree XSieve", ALE::Arg<int>().DEFAULT(3));
        argDB("treeFanout", "The fanout factor of the tree XSieve: number of children", ALE::Arg<int>().DEFAULT(3));
        argDB("pt", "The point to compute the boundary of", ALE::Arg<int>().DEFAULT(1));
        argDB("traversals", "The number of times to traverse the boundary", ALE::Arg<int>().DEFAULT(1));
      };// XSieveTester()
      //
      #undef __FUNCT__
      #define __FUNCT__ "createForkXSieve"
      static ALE::Obj<default_xsieve_type> createForkXSieve(const MPI_Comm& comm, const ALE::ArgDB& argDB) {
        typedef default_xsieve_type              xsieve_type;
        typedef xsieve_type::arrow_type          arrow_type;
        typedef std::set<arrow_type::target_type> RealBase;
        typedef std::set<arrow_type::source_type> RealCone;
        ALE::Obj<xsieve_type>   xsieve = new default_xsieve_type(comm);
        for(int i = 0; i < (int)argDB["baseSize"]; i++) {
          // Add an arrow from i mod baseSize to i with color 'Y'.
          xsieve->addArrow(arrow_type((double)(i%(int)argDB["capSize"]),i,'Y'));
        }
        return xsieve;
      };// createForkXSieve()
      //
      #undef __FUNCT__
      #define __FUNCT__ "createHatXSieve"
      static ALE::Obj<default_xsieve_type> createHatXSieve(const MPI_Comm& comm, const ALE::ArgDB& argDB) {
        typedef default_xsieve_type                      xsieve_type;
        typedef xsieve_type::arrow_type                  arrow_type;
        typedef std::set<arrow_type::target_type>         RealBase;
        typedef std::set<arrow_type::source_type>         RealCone;
        ALE::Obj<xsieve_type>   xsieve = new default_xsieve_type(comm);
        for(int i = 0; i < (int)argDB["baseSize"]; i++) {
          // Add an arrow from i to i mod baseSize with  color 'H'.
          xsieve->addArrow(arrow_type((double)i,i%(int)argDB["capSize"],'H'));
        }
        return xsieve;
      };// createHatXSieve()
      //
      #undef __FUNCT__
      #define __FUNCT__ "createTreeXSieve"
      static ALE::Obj<symmetric_xsieve_type> createTreeXSieve(const MPI_Comm& comm, const ALE::ArgDB& argDB) {
        typedef symmetric_xsieve_type             xsieve_type;
        typedef xsieve_type::arrow_type           arrow_type;
        typedef std::set<arrow_type::target_type> RealBase;
        typedef std::set<arrow_type::source_type> RealCone;
        ALE::Obj<xsieve_type>   tree = new xsieve_type(comm);
        int depth = argDB["treeDepth"], fanout = argDB["treeFanout"], tail = 0;
        _createSubtree(tree, depth, fanout, tail);
        return tree;
      };// createTreeXSieve()
      //
      // aux function: create a subtree; called recursively
      static int _createSubtree(ALE::Obj<symmetric_xsieve_type> tree, int depth, int fanout, int& tail){
        typedef symmetric_xsieve_type::arrow_type arrow_type;
        // tail is the end of the interval already used up in the tree construction
        int root = tail+1;
        tail = root;
        if(depth > 0) {
          for(int i = 0; i < fanout; ++i){
            // construct subtrees
            int subroot = _createSubtree(tree, depth-1, fanout, tail);
            // add arrows from each of the subroots to the root
            tree->addArrow(arrow_type(subroot, root, 'T'));
          }
        }
        return root;
      }// _createSubtree()
      //
      #undef __FUNCT__
      #define __FUNCT__ "BasicTest"
      template <typename XSieve_>
      static PetscErrorCode BasicTest(const ALE::Obj<XSieve_>& xsieve, const ALE::ArgDB& argDB, const char* xsieveName = NULL)
      {
        typedef XSieve_                                   xsieve_type;
        typedef typename xsieve_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;
        PetscFunctionBegin;
        ALE::LogStage stage = ALE::LogStageRegister("Basic Test");
        ALE::LogStagePush(stage);
        std::cout << "\nXSieve Basic Test:\n";
        if(xsieveName != NULL) {
          std::cout << xsieveName << ":\n";
        }
        std::cout << *xsieve << "\n";
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }// BasicTest()
      //
      #undef __FUNCT__
      #define __FUNCT__ "BaseRangeTest"
      template <typename XSieve_>
      static PetscErrorCode BaseRangeTest(const ALE::Obj<XSieve_>& xsieve, const ALE::ArgDB& argDB, const char* xsieveName = NULL)
      {
        typedef XSieve_                                   xsieve_type;
        typedef typename xsieve_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;
        ALE::Obj<typename xsieve_type::BaseSequence> base = xsieve->base(); 
        PetscFunctionBegin;
        ALE::LogStage stage = ALE::LogStageRegister("Base Test");
        ALE::LogStagePush(stage);
        std::cout << "\nXSieve Base Range Test:\n";
        if(xsieveName != NULL) {
          std::cout << xsieveName << ":\n";
        }
        typename xsieve_type::BaseSequence::iterator bbegin, bend;
        bbegin = base->begin();
        bend   = base->end();
        std::cout << "Base: *bbegin: " << *bbegin;
#ifndef ALE_XSIEVE_USE_ARROW_LINKS
        std::cout << ", *bend: " << *bend;
#endif
        std::cout << std::endl;
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }//BaseRangeTest()
      //
      #undef __FUNCT__
      #define __FUNCT__ "BaseTest"
      template <typename XSieve_>
      static PetscErrorCode BaseTest(const ALE::Obj<XSieve_>& xsieve, ALE::ArgDB argDB, const char* xsieveName = NULL)
      {        
        typedef XSieve_                                   xsieve_type;
        typedef typename xsieve_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;
        //
        bool silent = argDB["silent"];
        int  testCount = argDB["iterations"];
        //
        ALE::Obj<typename xsieve_type::BaseSequence> base;
        RealBase realBase;
        base = xsieve->base();
        PetscFunctionBegin;
        for(int i = 0; i < testCount; ++i) {
          if(!silent) {std::cout << "\nXSieve Base Test: iter: " << i << "\n";}
          ALE::LogStage stage = ALE::LogStageRegister("Base Test");
          ALE::LogStagePush(stage);
          if(!silent) {
            if(xsieveName != NULL) {
              std::cout << xsieveName << ":\n";
            }
            for(typename xsieve_type::iterator ii = xsieve->begin(); ii != xsieve->end(); ++ii){
              realBase.insert(ii->target());
            }
          }
          typename xsieve_type::BaseSequence::iterator bbegin, bend, iter;
          bbegin = base->begin();
          bend   = base->end();
          iter   = bbegin;
          if(!silent) {
            std::cout << "Base: " << std::endl;
            int counter = 0;
            for(; !(iter == bend); ++iter) {   
              std::cout << *iter << ", ";
              counter++;
            }
            std::cout << std::endl;
            std::cout << "realBase: " << std::endl;
            for(typename RealBase::iterator ii = realBase.begin(); ii!=realBase.end(); ++ii) {   
              std::cout << *ii << ", ";
            }
            std::cout << std::endl;
          }
          ALE::LogStagePop(stage);
        }//for(int i = 0; i < testCount; ++i)
        PetscFunctionReturn(0);
      }//BaseTest()
      //
      #undef __FUNCT__
      #define __FUNCT__ "ConeTest"
      template <typename XSieve_>
      static PetscErrorCode ConeTest(const ALE::Obj<XSieve_>& xsieve, const ALE::ArgDB& argDB, const char* xsieveName = NULL)
      {
        typedef XSieve_                                   xsieve_type;
        typedef typename xsieve_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;

        ALE::Obj<typename xsieve_type::BaseSequence> base = xsieve->base();
        typename xsieve_type::BaseSequence::iterator b_iter, b_next,
          b_begin = base->begin(), b_end = base->end();
        PetscFunctionBegin;
        int testCount = argDB["iterations"];
        bool silent = argDB["silent"];
        for(int i = 0; i < testCount; ++i) {
          ALE::LogStage stage = ALE::LogStageRegister("Cone Test");
          ALE::LogStagePush(stage);
          if(!silent){
            std::cout << "XSieve Cone Test: iter: " << i << "\n";
            if(xsieveName != NULL) {
              std::cout << xsieveName << ":\n";
            }
          }
          //
          b_iter  = b_begin;
          for(; b_iter != b_end; ++b_iter) {// base loop
            b_next = b_iter; ++b_next;
            // build cone
            ALE::Obj<typename xsieve_type::ConeSequence> cone;
            cone = xsieve->cone(*b_iter);
            if(!silent){
              // build realCone
              RealCone realCone;
              for(typename xsieve_type::iterator ii = xsieve->begin(); ii != xsieve->end(); ++ii){
                if(ii->target() == *b_iter) {
                  realCone.insert(ii->source());
                }
              }
              // display cone
              typename xsieve_type::ConeSequence::iterator c_begin, c_end, c_iter;
              c_begin = cone->begin();
              c_end   = cone->end();
              c_iter   = c_begin;
              std::cout << "    cone    (" << *b_iter << ")";
              std::cout << ": [";
              for(; c_iter != c_end; ) {
                std::cout << *c_iter << ", ";
                ++c_iter;
              }
              std::cout << "]" << std::endl;
              // display realCone
              std::cout << "    realCone(" << *b_iter << ")";
              std::cout << ": [";
              for(typename RealCone::iterator ii = realCone.begin(); ii!=realCone.end(); ++ii) {   
                std::cout << *ii << ", ";
              }
              std::cout << "] " << std::endl;
            }//if(!silent)
          }// base loop
          if(!silent){std::cout << std::endl;}            
          ALE::LogStagePop(stage);
        }// for(i=0; i < testCount; ++i)
        PetscFunctionReturn(0);
      }// ConeTest()
      //
      #undef __FUNCT__
      #define __FUNCT__ "SliceBasicTest"
      template <typename XSieve_>
      static PetscErrorCode SliceBasicTest(const ALE::Obj<XSieve_>& xsieve, const ALE::ArgDB& argDB, const char* xsieveName = NULL)
      {
        typedef XSieve_                                   xsieve_type;
        typedef typename xsieve_type::arrow_type          arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;
        PetscFunctionBegin;
        std::cout << "\nBasic Slice Test:\n";
        if(xsieveName != NULL) {
          std::cout << xsieveName << ":\n";
        }        
        ALE::LogStage stage = ALE::LogStageRegister("SliceBasicTest");
        ALE::LogStagePush(stage);
        {// first slice scope
          std::cout << "First slice scope:\n";
          try {
            string sliceName("'Total1' of ");
            if(xsieveName != NULL) {sliceName = sliceName + string(xsieveName) + string(" ");};
            string label;
            std::cout << "Attempting to allocate an ArrowSlice " << sliceName << "\n";
            typename xsieve_type::ArrowSlice slice = xsieve->slice();
            std::cout << "No problem!\n";
            //
            label = sliceName + string(" unpopulated");
            slice.view(std::cout, label.c_str());
            //// populating slice
            // traverse the whole sifter and insert the arrows into the slice
            typename xsieve_type::ArrowSlice::marker_type marker = argDB["marker"];
            for(typename xsieve_type::iterator iter = xsieve->begin(); iter != xsieve->end(); ++iter) {
              slice.add(*iter, marker);
            }
            label = sliceName + string("populated");
            slice.view(std::cout, label.c_str());
            std::cout << "xsieve state:\n" << *xsieve << "\n";
            //
            slice.clean();
            label = sliceName + string("cleaned");
            slice.view(std::cout, label.c_str());
            std::cout << "xsieve state:\n";
            xsieve->view(std::cout);
            //
            std::cout << "Attempting to allocate another slice:\n";
            xsieve->slice();
            std::cout << "No problem!\n";
          } catch(ALE::XSieveDef::NoSlices e) {
            std::cout << "Caught a 'NoSlices' exception\n";
          }
        }// first slice scope
        {
          std::cout << "Second slice scope:\n"; 
          try {
            string sliceName("'Total2' of ");
            if(xsieveName != NULL) {sliceName = sliceName + string(xsieveName) + string(" ");};
            string label;
            std::cout << "Attempting to allocate an ArrowSlice " << sliceName << "\n";
            typename xsieve_type::ArrowSlice slice(xsieve->slice());
            std::cout << "No problem!\n";
            label = sliceName + string(" unpopulated");
            slice.view(std::cout, label.c_str());
            std::cout << "Attempting to allocate another slice:\n";
            xsieve->slice();
            std::cout << "No problem!\n";
          }
          catch(ALE::XSieveDef::NoSlices e) {
            std::cout << "Caught a 'NoSlices' exception\n";
          }
        }// second slice scope
        ALE::LogStagePop(stage);
        PetscFunctionReturn(0);
      }// SliceBasicTest()
     //
      #undef __FUNCT__
      #define __FUNCT__ "BoundaryTest"
      template <typename XSieve_>
      static PetscErrorCode BoundaryTest(const ALE::Obj<XSieve_>& xsieve, ALE::ArgDB argDB, const char* xsieveName = NULL)
      {        
        typedef XSieve_                                    xsieve_type;
        typedef typename xsieve_type::arrow_type           arrow_type;
        typedef std::set<typename arrow_type::target_type> RealBase;
        typedef std::set<typename arrow_type::source_type> RealCone;
        typedef std::set<typename arrow_type::source_type> RealClosure;

        PetscFunctionBegin;

        ALE::Obj<typename xsieve_type::BaseSequence> base;
        typename arrow_type::target_type pt = argDB["pt"];
        bool silent = argDB["silent"];
        int  testCount = argDB["iterations"];
        int  traversalCount = argDB["traversals"];
        for(int i = 0; i < testCount; ++i){
          if(!silent){std::cout << "\nXSieve Boundary Test: iter: " << i << "\n";}
          // Boundary Slice
          {
            if(!silent) {std::cout << "Slice version\n";}
            static ALE::NoOp<arrow_type> noop;
            ALE::LogStage stage = ALE::LogStageRegister("Boundary Slice Test");
            ALE::LogStagePush(stage);
            string label;
            if(!silent) {
              if(xsieveName != NULL) {
                label = string(xsieveName) + ": ";
              }        
              else {
                label = string(": ");
              }
            }
            if(!silent) {
              std::cout << label << "before taking boundary:\n" << (*xsieve) << "\n";
            }
            typename xsieve_type::BoundarySlice bd = xsieve->boundarySlice(pt);
            if(!silent) {
              std::cout << "Slice bd(" << pt << ")= " << bd << "\n";
              std::cout << "\n";
              std::cout << label << "after taking boundary:\n" << (*xsieve) << "\n";
            }
            for(int i = 0; i < traversalCount;++i) {
              bd.traverse(noop);
            }
            ALE::LogStagePop(stage);
          }
          // Boundary Set
          {
            if(!silent){std::cout << "Set version\n";}
            static ALE::NoOp<typename arrow_type::source_type> noop;
            ALE::LogStage stage = ALE::LogStageRegister("Boundary Set Test");
            ALE::LogStagePush(stage);
            string label;
            if(!silent) {
              if(xsieveName != NULL) {
                label = string(xsieveName) + ": ";
              }        
              else {
                label = string(": ");
              }
            }
            typename xsieve_type::BoundarySet bd = xsieve->boundarySet(pt);
            if(!silent) {
              std::cout << "Set bd(" << pt << ")= " << bd << "\n";
            }
            for(int i = 0; i < traversalCount;++i) {
              bd.traverse(noop);
            }
            ALE::LogStagePop(stage);
          }
        }// for(..., i < testCount; ...)
        //
        PetscFunctionReturn(0);
      }// BoundaryTest()
      //
    };// struct XSieveTester
  };//namespace Test
};// namespace ALE

#endif
