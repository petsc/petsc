static char help[] = "Basic ArgDB functionality test.\n\n";

#include <ALE.hh>
#include <iostream>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv){
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);

  // Basic usage of ArgDB/Arg<T>:
  // (1) declare db
  //   ArgDB argDB("Comment whole db");
  // (2) add args descriptions
  //   argDB("arg1", "arg1 help", Arg<T1>.DEFAULT(d1));
  //   argDB("arg2", "arg2 help", Arg<T2>.IS_A_LIST.DEFAULT(d2));
  //   argDB("arg3", "arg3 help", Arg<T3>.IS_A_FLAG.DEFAULT(false));
  //   argDB("arg4", "arg4 help", Arg<T4>.IS_MULTIPLACED.DEFAULT(d4)); // may be defined multiple times at different places on the command line
  // (3) parse command-line args
  //   argDB.parse(argc,argv)
  // (4) retrieve args
  //   T1 t1 = argDB["arg1"]; 
  //   T  t  = argDB["arg2"]; // ok if T2 can be cast to T, otherwise, an error
  try{
    ALE::ArgDB argDB("General options");
    argDB("debug", "debugging level", ALE::Arg<int>().DEFAULT(0));
    std::cout << argDB << "\n";
    argDB.parse(argc,argv);
    int    idebug = argDB["debug"];
    //double ddebug = argDB["debug"]; // error
    //double ddebug = argDB["debug"].as<double>(); // error
    std::cout << "(int)    debug = " << idebug << "\n";
  }
  catch(ALE::Exception& e) {
    std::cout << "Caught exception: " << e << "\n";
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
