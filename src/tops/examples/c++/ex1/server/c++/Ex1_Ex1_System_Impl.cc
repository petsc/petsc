// 
// File:          Ex1_Ex1_System_Impl.cc
// Symbol:        Ex1.Ex1_System-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.2
// Description:   Server-side implementation for Ex1.Ex1_System
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.2
// 
#include "Ex1_Ex1_System_Impl.hh"

// DO-NOT-DELETE splicer.begin(Ex1.Ex1_System._includes)
// Insert-Code-Here {Ex1.Ex1_System._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(Ex1.Ex1_System._includes)

// user-defined constructor.
void Ex1::Ex1_System_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System._ctor)
  // Insert-Code-Here {Ex1.Ex1_System._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System._ctor)
}

// user-defined destructor.
void Ex1::Ex1_System_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System._dtor)
  // Insert-Code-Here {Ex1.Ex1_System._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System._dtor)
}

// static class initializer.
void Ex1::Ex1_System_impl::_load() {
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System._load)
  // Insert-Code-Here {Ex1.Ex1_System._load} (class initialization)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  setSolver[]
 */
void
Ex1::Ex1_System_impl::setSolver (
  /* in */ ::TOPS::Solver solver ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System.setSolver)
  // Insert-Code-Here {Ex1.Ex1_System.setSolver} (setSolver method)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System.setSolver)
}

/**
 * Method:  initializeOnce[]
 */
void
Ex1::Ex1_System_impl::initializeOnce ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System.initializeOnce)
  // Insert-Code-Here {Ex1.Ex1_System.initializeOnce} (initializeOnce method)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System.initializeOnce)
}

/**
 * Method:  initializeEverySolve[]
 */
void
Ex1::Ex1_System_impl::initializeEverySolve ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System.initializeEverySolve)
  // Insert-Code-Here {Ex1.Ex1_System.initializeEverySolve} (initializeEverySolve method)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System.initializeEverySolve)
}

/**
 * Method:  computeJacobian[]
 */
void
Ex1::Ex1_System_impl::computeJacobian (
  /* in */ ::sidl::array<double> x,
  /* in */ ::TOPS::Matrix J,
  /* in */ ::TOPS::Matrix B ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System.computeJacobian)
  // Insert-Code-Here {Ex1.Ex1_System.computeJacobian} (computeJacobian method)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System.computeJacobian)
}

/**
 * Method:  computeResidual[]
 */
void
Ex1::Ex1_System_impl::computeResidual (
  /* in */ ::sidl::array<double> x,
  /* in */ ::sidl::array<double> f ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System.computeResidual)
  // Insert-Code-Here {Ex1.Ex1_System.computeResidual} (computeResidual method)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System.computeResidual)
}

/**
 * Method:  computeRightHandSide[]
 */
void
Ex1::Ex1_System_impl::computeRightHandSide (
  /* in */ ::sidl::array<double> b ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System.computeRightHandSide)
  // Insert-Code-Here {Ex1.Ex1_System.computeRightHandSide} (computeRightHandSide method)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System.computeRightHandSide)
}

/**
 * Method:  computeInitialGuess[]
 */
void
Ex1::Ex1_System_impl::computeInitialGuess (
  /* in */ ::sidl::array<double> x ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(Ex1.Ex1_System.computeInitialGuess)
  // Insert-Code-Here {Ex1.Ex1_System.computeInitialGuess} (computeInitialGuess method)
  // DO-NOT-DELETE splicer.end(Ex1.Ex1_System.computeInitialGuess)
}


// DO-NOT-DELETE splicer.begin(Ex1.Ex1_System._misc)
// Insert-Code-Here {Ex1.Ex1_System._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(Ex1.Ex1_System._misc)

