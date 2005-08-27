// 
// File:          TOPS_Solver_Sliced_Impl.cc
// Symbol:        TOPS.Solver_Sliced-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.Solver_Sliced
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 
#include "TOPS_Solver_Sliced_Impl.hh"

// DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced._includes)
// Insert-Code-Here {TOPS.Solver_Sliced._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced._includes)

// user-defined constructor.
void TOPS::Solver_Sliced_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced._ctor)
  // Insert-Code-Here {TOPS.Solver_Sliced._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced._ctor)
}

// user-defined destructor.
void TOPS::Solver_Sliced_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced._dtor)
  // Insert-Code-Here {TOPS.Solver_Sliced._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced._dtor)
}

// static class initializer.
void TOPS::Solver_Sliced_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced._load)
  // Insert-Code-Here {TOPS.Solver_Sliced._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setSystem[]
 */
void
TOPS::Solver_Sliced_impl::setSystem (
  /* in */ ::TOPS::System::System system ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setSystem)
  // Insert-Code-Here {TOPS.Solver_Sliced.setSystem} (setSystem method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setSystem)
}

/**
 * Method:  getSystem[]
 */
::TOPS::System::System
TOPS::Solver_Sliced_impl::getSystem ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.getSystem)
  // Insert-Code-Here {TOPS.Solver_Sliced.getSystem} (getSystem method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.getSystem)
}

/**
 * Method:  Initialize[]
 */
void
TOPS::Solver_Sliced_impl::Initialize (
  /* in */ ::sidl::array< ::std::string> args ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.Initialize)
  // Insert-Code-Here {TOPS.Solver_Sliced.Initialize} (Initialize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.Initialize)
}

/**
 * Method:  solve[]
 */
void
TOPS::Solver_Sliced_impl::solve ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.solve)
  // Insert-Code-Here {TOPS.Solver_Sliced.solve} (solve method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.solve)
}

/**
 * Method:  setBlockSize[]
 */
void
TOPS::Solver_Sliced_impl::setBlockSize (
  /* in */ int32_t bs ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setBlockSize)
  // Insert-Code-Here {TOPS.Solver_Sliced.setBlockSize} (setBlockSize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setBlockSize)
}

/**
 * Method:  getSolution[]
 */
::sidl::array<double>
TOPS::Solver_Sliced_impl::getSolution ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.getSolution)
  // Insert-Code-Here {TOPS.Solver_Sliced.getSolution} (getSolution method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.getSolution)
}

/**
 * Method:  setSolution[]
 */
void
TOPS::Solver_Sliced_impl::setSolution (
  /* in */ ::sidl::array<double> location ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setSolution)
  // Insert-Code-Here {TOPS.Solver_Sliced.setSolution} (setSolution method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setSolution)
}

/**
 * Method:  setLocalRowSize[]
 */
void
TOPS::Solver_Sliced_impl::setLocalRowSize (
  /* in */ int32_t m ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setLocalRowSize)
  // Insert-Code-Here {TOPS.Solver_Sliced.setLocalRowSize} (setLocalRowSize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setLocalRowSize)
}

/**
 * Method:  getLocalRowSize[]
 */
int32_t
TOPS::Solver_Sliced_impl::getLocalRowSize ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.getLocalRowSize)
  // Insert-Code-Here {TOPS.Solver_Sliced.getLocalRowSize} (getLocalRowSize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.getLocalRowSize)
}

/**
 * Method:  setGlobalRowSize[]
 */
void
TOPS::Solver_Sliced_impl::setGlobalRowSize (
  /* in */ int32_t M ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setGlobalRowSize)
  // Insert-Code-Here {TOPS.Solver_Sliced.setGlobalRowSize} (setGlobalRowSize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setGlobalRowSize)
}

/**
 * Method:  getGlobalRowSize[]
 */
int32_t
TOPS::Solver_Sliced_impl::getGlobalRowSize ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.getGlobalRowSize)
  // Insert-Code-Here {TOPS.Solver_Sliced.getGlobalRowSize} (getGlobalRowSize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.getGlobalRowSize)
}

/**
 * Method:  setLocalColumnSize[]
 */
void
TOPS::Solver_Sliced_impl::setLocalColumnSize (
  /* in */ int32_t n ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setLocalColumnSize)
  // Insert-Code-Here {TOPS.Solver_Sliced.setLocalColumnSize} (setLocalColumnSize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setLocalColumnSize)
}

/**
 * Method:  getLocalColumnSize[]
 */
int32_t
TOPS::Solver_Sliced_impl::getLocalColumnSize ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.getLocalColumnSize)
  // Insert-Code-Here {TOPS.Solver_Sliced.getLocalColumnSize} (getLocalColumnSize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.getLocalColumnSize)
}

/**
 * Method:  setGlobalColumnSize[]
 */
void
TOPS::Solver_Sliced_impl::setGlobalColumnSize (
  /* in */ int32_t N ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setGlobalColumnSize)
  // Insert-Code-Here {TOPS.Solver_Sliced.setGlobalColumnSize} (setGlobalColumnSize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setGlobalColumnSize)
}

/**
 * Method:  getGlobalColumnSize[]
 */
int32_t
TOPS::Solver_Sliced_impl::getGlobalColumnSize ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.getGlobalColumnSize)
  // Insert-Code-Here {TOPS.Solver_Sliced.getGlobalColumnSize} (getGlobalColumnSize method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.getGlobalColumnSize)
}

/**
 * Method:  setGhostPoints[]
 */
void
TOPS::Solver_Sliced_impl::setGhostPoints (
  /* in */ ::sidl::array<int32_t> ghosts ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setGhostPoints)
  // Insert-Code-Here {TOPS.Solver_Sliced.setGhostPoints} (setGhostPoints method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setGhostPoints)
}

/**
 * Method:  setPreallocation[]
 */
void
TOPS::Solver_Sliced_impl::setPreallocation (
  /* in */ int32_t d,
  /* in */ int32_t od ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setPreallocation)
  // Insert-Code-Here {TOPS.Solver_Sliced.setPreallocation} (setPreallocation method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setPreallocation)
}

/**
 * Method:  setPreallocation[s]
 */
void
TOPS::Solver_Sliced_impl::setPreallocation (
  /* in */ ::sidl::array<int32_t> d,
  /* in */ ::sidl::array<int32_t> od ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setPreallocations)
  // Insert-Code-Here {TOPS.Solver_Sliced.setPreallocations} (setPreallocation method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setPreallocations)
}

/**
 * Starts up a component presence in the calling framework.
 * @param services the component instance's handle on the framework world.
 * Contracts concerning Svc and setServices:
 * 
 * The component interaction with the CCA framework
 * and Ports begins on the call to setServices by the framework.
 * 
 * This function is called exactly once for each instance created
 * by the framework.
 * 
 * The argument Svc will never be nil/null.
 * 
 * Those uses ports which are automatically connected by the framework
 * (so-called service-ports) may be obtained via getPort during
 * setServices.
 */
void
TOPS::Solver_Sliced_impl::setServices (
  /* in */ ::gov::cca::Services services ) 
throw ( 
  ::gov::cca::CCAException
){
  // DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced.setServices)
  // Insert-Code-Here {TOPS.Solver_Sliced.setServices} (setServices method)
  // DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced.setServices)
}


// DO-NOT-DELETE splicer.begin(TOPS.Solver_Sliced._misc)
// Insert-Code-Here {TOPS.Solver_Sliced._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.Solver_Sliced._misc)

