#ifndef included_ALE_xsifterTest_hh
#define included_ALE_xsifterTest_hh

#include <XSifter.hh>

namespace ALE_X {
  namespace Test {
    typedef ALE_X::SifterDef::Arrow<double,int,char>    arrow_type;
    typedef ALE_X::Sifter<arrow_type>                   sifter_type;
    class SifterTest {
    public:
      static ALE::Obj<sifter_type> createForkSifter(const int capSize = 10, const int debug = 0) {
        ALE::Obj<sifter_type>   sifter = new sifter_type();
        for(int i = 0; i < capSize; i++) {
          // Add an arrow from i to i mod 3 with color 'X'.
          sifter->addArrow(arrow_type((double)i,i % 3,'X'));
        }
        return sifter;
      };
    };// class SifterTest

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
        
        ierr = PetscOptionsBegin(comm, "", "Options for sifter basic test", "Sieve");CHKERRQ(ierr);
        ierr = PetscOptionsInt("-debug", "The debugging level", "xsifter0.cxx", 0, &options->debug, PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "xsifter0.cxx", options->iters, &options->iters, 
                               PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();
        PetscFunctionReturn(0);
      }
    };
  };// namespace Test
};// namespace ALE_X

#endif
