#ifndef included_ALE_xsifterTest_hh
#define included_ALE_xsifterTest_hh

#include <petsc.h>
#include <XSifter.hh>

namespace ALE {
  namespace Test {
    namespace XSifter {
      typedef ALE::XSifterDef::Arrow<double,int,char>    arrow_type;
      typedef ALE::ALE_XSIFTER_TYPE<arrow_type>          xsifter_type;
      typedef std::set<arrow_type::target_type>          RealBase;
      typedef std::set<arrow_type::source_type>          RealCone;
      //
      struct Options {
        int      debug;   // The debugging level
        int      codebug; // The codebugging level
        PetscInt iters;   // The number of test repetitions
        PetscInt capSize; // The size of the Fork sifter cap
        PetscInt baseSize; // The size of the Fork sifter base
        PetscInt universeSize; // The size of the predicate universe: [0,universeSize).
        PetscInt predicate; // The slice predicate; negative means 'all arrows'
        Options(MPI_Comm comm = PETSC_COMM_WORLD){
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
          options->debug     = 0;
          options->codebug   = 0;
          options->iters     = 1;
          options->capSize   = 3;
          options->baseSize  = 10;
          options->universeSize = 2;
          options->predicate = -1;
          ierr = PetscOptionsBegin(comm, "", "Options for xsifter basic test", "XSifter");CHKERRQ(ierr);
          ierr = PetscOptionsInt("-debug",   "The debugging level", "xsifter0.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsInt("-codebug", "The co-debugging level", "xsifter0.cxx", options->codebug, &options->codebug, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsInt("-iterations","The number of test repetitions", "xsifter0.cxx", options->iters, &options->iters, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsInt("-capSize", "The size of Fork xsifter cap", "xsifter0.cxx", options->capSize, &options->capSize, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsInt("-baseSize", "The size of Fork xsifter base", "xsifter0.cxx", options->baseSize, &options->baseSize, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsInt("-universeSize", "The size of predicate universe", "xsifter0.cxx", options->universeSize, &options->universeSize, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsInt("-predicate", "The slice predicate (negative implies 'all'", "xsifter0.cxx", options->predicate, &options->predicate, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsEnd();
          //
          ALE::XSifterDef::debug   = options->debug;
          ALE::XSifterDef::codebug = options->codebug;
          PetscFunctionReturn(0);
        }
      };// struct Options
      //
      static ALE::Obj<xsifter_type> createForkXSifter(const MPI_Comm& comm, const Options& options) {
        ALE::Obj<xsifter_type>   xsifter = new xsifter_type(comm, options.debug);
        for(int i = 0; i < options.baseSize; i++) {
          // Add an arrow from i mod baseSize to i with predicate (i+1) % universeSize and color 'Y'.
          xsifter->addArrow(arrow_type((double)(i%options.capSize),i,'Y'),(i+1)%options.universeSize);
        }
        return xsifter;
      };// createForkXSifter()
      static ALE::Obj<xsifter_type> createHatXSifter(const MPI_Comm& comm, const Options& options) {
        ALE::Obj<xsifter_type>   xsifter = new xsifter_type(comm, options.debug);
        for(int i = 0; i < options.baseSize; i++) {
          // Add an arrow from i to i mod baseSize with predicate (i+2) % universeSize and color 'H'.
          xsifter->addArrow(arrow_type((double)i,i%options.capSize,'H'),(i+2)%options.universeSize);
        }
        return xsifter;
      };// createHatXSifter()
    };//namespace XSifter
  };// namespace Test
};// namespace ALE

#endif
