#ifndef included_ALE_xsifterTest_hh
#define included_ALE_xsifterTest_hh

#include <petsc.h>
#include <XSifter.hh>

namespace ALE {
  namespace Test {
    typedef ALE::XSifterDef::Arrow<double,int,char>    arrow_type;
    typedef ALE::XSifter<arrow_type>                   xsifter_type;
    class XSifterTest {
    public:
      static ALE::Obj<xsifter_type> createForkXSifter(MPI_Comm comm, const int capSize = 10, const int debug = 0) {
        ALE::Obj<xsifter_type>   xsifter = new xsifter_type();
        for(int i = 0; i < capSize; i++) {
          // Add an arrow from i to i mod 3 with color 'X'.
          xsifter->addArrow(arrow_type((double)i,i % 3,'X'));
        }
        return xsifter;
      };
    };// class XSifterTest

    struct Options {
      int      debug; // The debugging level
      PetscInt iters; // The number of test repetitions
      Options(MPI_Comm comm = PETSC_COMM_SELF){
        PetscErrorCode ierr = ProcessOptions(comm, this); 
        ALE::CHKERROR(ierr, "Error in Options constructor/ProcessOptions");
      };
      //
      #undef  __FUNCT__
      #define __FUNCT__ "ProcessOptions"
      PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
      {
        PetscErrorCode ierr;
        
        PetscFunctionBegin;
        options->debug = debug;
        options->iters = 1;
        
        ierr = PetscOptionsBegin(comm, "", "Options for xsifter basic test", "XSifter");CHKERRQ(ierr);
        ierr = PetscOptionsInt("-debug", "The debugging level", "xsifter0.cxx", 0, &options->debug, PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "xsifter0.cxx", options->iters, &options->iters, 
                               PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();
        PetscFunctionReturn(0);
      }
    };
  };// namespace Test
};// namespace ALE

#endif
