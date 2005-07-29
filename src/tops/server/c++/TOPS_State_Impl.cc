// 
// File:          TOPS_State_Impl.cc
// Symbol:        TOPS.State-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for TOPS.State
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "TOPS_State_Impl.hh"

// DO-NOT-DELETE splicer.begin(TOPS.State._includes)
#include "petsc.h"
// DO-NOT-DELETE splicer.end(TOPS.State._includes)

// user-defined constructor.
void TOPS::State_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.State._ctor)
  // Insert-Code-Here {TOPS.State._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(TOPS.State._ctor)
}

// user-defined destructor.
void TOPS::State_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.State._dtor)
  PetscTruth finalized;
  int ierr = PetscFinalized(&finalized);
  if (!finalized) {
    ierr = PetscFinalize();
  }
  // DO-NOT-DELETE splicer.end(TOPS.State._dtor)
}

// static class initializer.
void TOPS::State_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.State._load)
  // Insert-Code-Here {TOPS.State._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.State._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  Initialize[]
 */
void
TOPS::State_impl::Initialize (
  /* in */ ::sidl::array< ::std::string> args ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.State.Initialize)
  int          argc = args.upper(0) + 1;
  char       **argv = new char* [argc];
  std::string  arg;

  for(int i = 0; i < argc; i++) {
    arg     = args[i];
    argv[i] = new char [arg.length()+1];
    arg.copy(argv[i], arg.length(), 0);
    argv[i][arg.length()] = 0;
  }
  int    ierr = PetscInitialize(&argc,&argv,0,0);
  // DO-NOT-DELETE splicer.end(TOPS.State.Initialize)
}

/**
 * Method:  Finalize[]
 */
void
TOPS::State_impl::Finalize ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.State.Finalize)
  // Insert-Code-Here {TOPS.State.Finalize} (Finalize method)
  // DO-NOT-DELETE splicer.end(TOPS.State.Finalize)
}


// DO-NOT-DELETE splicer.begin(TOPS.State._misc)
// Insert-Code-Here {TOPS.State._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.State._misc)

