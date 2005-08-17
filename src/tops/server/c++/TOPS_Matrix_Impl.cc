// 
// File:          TOPS_Matrix_Impl.cc
// Symbol:        TOPS.Matrix-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.Matrix
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 
#include "TOPS_Matrix_Impl.hh"

// DO-NOT-DELETE splicer.begin(TOPS.Matrix._includes)
// Insert-Code-Here {TOPS.Matrix._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(TOPS.Matrix._includes)

// user-defined constructor.
void TOPS::Matrix_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Matrix._ctor)
  this->mat = 0;
  // DO-NOT-DELETE splicer.end(TOPS.Matrix._ctor)
}

// user-defined destructor.
void TOPS::Matrix_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Matrix._dtor)
  // Insert-Code-Here {TOPS.Matrix._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(TOPS.Matrix._dtor)
}

// static class initializer.
void TOPS::Matrix_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.Matrix._load)
  // Insert-Code-Here {TOPS.Matrix._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.Matrix._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  apply[]
 */
void
TOPS::Matrix_impl::apply (
  /* in */ ::TOPS::Vector x,
  /* in */ ::TOPS::Vector y ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Matrix.apply)
  // Insert-Code-Here {TOPS.Matrix.apply} (apply method)
  // DO-NOT-DELETE splicer.end(TOPS.Matrix.apply)
}

/**
 * Method:  zero[]
 */
void
TOPS::Matrix_impl::zero ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Matrix.zero)
  // Insert-Code-Here {TOPS.Matrix.zero} (zero method)
  // DO-NOT-DELETE splicer.end(TOPS.Matrix.zero)
}

/**
 * Method:  set[D2]
 */
void
TOPS::Matrix_impl::set (
  /* in */ int32_t i,
  /* in */ int32_t j,
  /* in */ ::sidl::array<double> values ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Matrix.setD2)
  // Insert-Code-Here {TOPS.Matrix.setD2} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.Matrix.setD2)
}


// DO-NOT-DELETE splicer.begin(TOPS.Matrix._misc)
// Insert-Code-Here {TOPS.Matrix._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.Matrix._misc)

