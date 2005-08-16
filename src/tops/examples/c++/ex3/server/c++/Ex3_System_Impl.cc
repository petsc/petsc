// 
// File:          Ex3_System_Impl.cc
// Symbol:        Ex3.System-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for Ex3.System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "Ex3_System_Impl.hh"

// DO-NOT-DELETE splicer.begin(Ex3.System._includes)
// Insert-Code-Here {Ex3.System._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(Ex3.System._includes)

// user-defined constructor.
void Ex3::System_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex3.System._ctor)
  // Insert-Code-Here {Ex3.System._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(Ex3.System._ctor)
}

// user-defined destructor.
void Ex3::System_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex3.System._dtor)
  // Insert-Code-Here {Ex3.System._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex3.System._dtor)
}

// static class initializer.
void Ex3::System_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex3.System._load)
  // Insert-Code-Here {Ex3.System._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex3.System._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  computeJacobian[]
 */
void
Ex3::System_impl::computeJacobian (
  /* in */ ::sidl::array<double> x,
  /* in */ ::TOPS::Matrix J,
  /* in */ ::TOPS::Matrix B ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex3.System.computeJacobian)
  // Insert-Code-Here {Ex3.System.computeJacobian} (computeJacobian method)
  // DO-NOT-DELETE splicer.end(Ex3.System.computeJacobian)
}

/**
 * Method:  setSolver[]
 */
void
Ex3::System_impl::setSolver (
  /* in */ ::TOPS::Solver solver ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex3.System.setSolver)
  // Insert-Code-Here {Ex3.System.setSolver} (setSolver method)
  // DO-NOT-DELETE splicer.end(Ex3.System.setSolver)
}

/**
 * Method:  initializeOnce[]
 */
void
Ex3::System_impl::initializeOnce ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex3.System.initializeOnce)
  // Insert-Code-Here {Ex3.System.initializeOnce} (initializeOnce method)
  // DO-NOT-DELETE splicer.end(Ex3.System.initializeOnce)
}

/**
 * Method:  initializeEverySolve[]
 */
void
Ex3::System_impl::initializeEverySolve ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex3.System.initializeEverySolve)
  // Insert-Code-Here {Ex3.System.initializeEverySolve} (initializeEverySolve method)
  // DO-NOT-DELETE splicer.end(Ex3.System.initializeEverySolve)
}

/**
 * Method:  computeRightHandSide[]
 */
void
Ex3::System_impl::computeRightHandSide (
  /* in */ ::sidl::array<double> b ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex3.System.computeRightHandSide)
  // Insert-Code-Here {Ex3.System.computeRightHandSide} (computeRightHandSide method)
  // DO-NOT-DELETE splicer.end(Ex3.System.computeRightHandSide)
}


// DO-NOT-DELETE splicer.begin(Ex3.System._misc)
// Insert-Code-Here {Ex3.System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex3.System._misc)

