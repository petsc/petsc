#ifndef included_ALE_xsieveTest_hh
#define included_ALE_xsieveTest_hh

#include <XSieve.hh>
#include <xsifterTest.hh>

namespace ALE {
  namespace Test {
    struct XSieveTester : public ::ALE::Test::XSifterTester {
      typedef ALE::Test::XSifterTester                xsifter_tester_type;
      typedef ALE::XSifterDef::Arrow<int,int,char>    default_arrow_type;
      typedef ALE::XSieve<default_arrow_type>         default_xsieve_type;
      //
      ALE::Component::ArgDB argDB;
      XSieveTester() : argDB("XSieve basic test options"){
        argDB(xsifter_tester_type::argDB);
        argDB("treeDepth", "The depth of the tree XSieve", ALE::Component::Arg<int>().DEFAULT(3));
        argDB("treeFanout", "The fanout factor of the tree XSieve: number of children", ALE::Component::Arg<int>().DEFAULT(3));
        argDB("pt", "The point to compute closure over", ALE::Component::Arg<int>().DEFAULT(1));
      };
      //
      #undef __FUNCT__
      #define __FUNCT__ "createTreeXSieve"
      static ALE::Obj<default_xsieve_type> createTreeXSieve(const MPI_Comm& comm, const ALE::Component::ArgDB& argDB) {
        typedef default_xsieve_type               xsieve_type;
        typedef xsieve_type::arrow_type           arrow_type;
        typedef std::set<arrow_type::target_type> RealBase;
        typedef std::set<arrow_type::source_type> RealCone;
        ALE::Obj<xsieve_type>   tree = new xsieve_type(comm, argDB["debug"]);
        int depth = argDB["treeDepth"], fanout = argDB["treeFanout"], tail = 0;
        _createSubtree(tree, depth, fanout, tail);
        return tree;
      };// createTreeXSieve()
      //
      // aux function: create a subtree; called recursively
      static int _createSubtree(ALE::Obj<default_xsieve_type> tree, int depth, int fanout, int& tail){
        typedef default_xsieve_type::arrow_type arrow_type;
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
      //
      #undef __FUNCT__
      #define __FUNCT__ "BoundaryTest"
      template <typename XSieve_>
      static PetscErrorCode BoundaryTest(const ALE::Obj<XSieve_>& xsieve, ALE::Component::ArgDB argDB, const char* xsieveName = NULL)
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
        for(int i = 0; i < testCount; ++i){
          if(!silent){std::cout << "XSieve Boundary Test: iter: " << i << "\n";}
          // Boundary Slice
          {
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
              std::cout << label << "before taking boundary\n";
              xsieve->view(std::cout);
            }
            typename xsieve_type::BoundarySlice bd = xsieve->boundarySlice(pt);
            if(!silent) {
              std::cout << "bd(" << pt << ")= ";
              bd.view(std::cout);
              std::cout << "\n";
              std::cout << label << "after taking boundary\n";
              xsieve->view(std::cout);
            }
            ALE::LogStagePop(stage);
          }
          // Boundary Set
          {
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
              std::cout << "bd(" << pt << ")= ";
              bd.view(std::cout);
              std::cout << "\n";
            }
            ALE::LogStagePop(stage);
          }
        }// for(..., i < testCount; ...)
        //
        PetscFunctionReturn(0);
      }// BoundaryTest()
    };// struct XSieveTester
  };//namespace Test
};// namespace ALE

#endif
