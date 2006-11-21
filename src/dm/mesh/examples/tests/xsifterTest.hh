#ifndef included_ALE_xsifterTest_hh
#define included_ALE_xsifterTest_hh

#include <petsc.h>
#include <XSifter.hh>

namespace ALE {
  namespace Test {
    namespace XSifter {
      typedef ALE::XSifterDef::Arrow<double,int,char>    arrow_type;
      typedef ALE::XSifter<arrow_type>                   xsifter_type;
      //
      struct Options {
        int      debug;   // The debugging level
        PetscInt iters;   // The number of test repetitions
        PetscInt capSize; // The size of the sifter cap
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
          options->debug = 0;
          options->iters = 1;
          options->capSize = 10;
          
          ierr = PetscOptionsBegin(comm, "", "Options for xsifter basic test", "XSifter");CHKERRQ(ierr);
          ierr = PetscOptionsInt("-debug", "The debugging level", "xsifter0.cxx", 0, &options->debug, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsInt("-iterations","The number of test repetitions", "xsifter0.cxx", options->iters, &options->iters, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsInt("-capSize", "The size of xsifter cap", "xsifter0.cxx", options->iters, &options->capSize, PETSC_NULL);CHKERRQ(ierr);
          ierr = PetscOptionsEnd();
          //
          ALE::XSifterDef::debug = options->debug;
          PetscFunctionReturn(0);
        }
      };// struct Options
      //
      static ALE::Obj<xsifter_type> createForkXSifter(const MPI_Comm& comm, const Options& options) {
        ALE::Obj<xsifter_type>   xsifter = new xsifter_type(comm, options.debug);
        for(int i = 0; i < options.capSize; i++) {
          // Add an arrow from i to i mod 3 with color 'Y'.
          xsifter->addArrow(arrow_type((double)i,i % 3,'Y'));
        }
        return xsifter;
      };// createForkXSifter()
      static ALE::Obj<xsifter_type> createHatXSifter(const MPI_Comm& comm, const Options& options) {
        ALE::Obj<xsifter_type>   xsifter = new xsifter_type(comm, options.debug);
        for(int i = 0; i < options.capSize; i++) {
          // Add an arrow from i mod 3 to i with color 'H'.
          xsifter->addArrow(arrow_type((double)(i % 3),i,'H'));
        }
        return xsifter;
      };// createHatXSifter()
    };//namespace XSifter
  };// namespace Test
};// namespace ALE

#endif
