// 
// File:          TOPS_MatrixStructured_Impl.cc
// Symbol:        TOPS.MatrixStructured-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.8
// Description:   Server-side implementation for TOPS.MatrixStructured
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.8
// 
#include "TOPS_MatrixStructured_Impl.hh"

// DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured._includes)
// Insert-Code-Here {TOPS.MatrixStructured._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(TOPS.MatrixStructured._includes)

// user-defined constructor.
void TOPS::MatrixStructured_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured._ctor)
  // Insert-Code-Here {TOPS.MatrixStructured._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured._ctor)
}

// user-defined destructor.
void TOPS::MatrixStructured_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured._dtor)
  // Insert-Code-Here {TOPS.MatrixStructured._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured._dtor)
}

// static class initializer.
void TOPS::MatrixStructured_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured._load)
  // Insert-Code-Here {TOPS.MatrixStructured._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  dimen[]
 */
int32_t
TOPS::MatrixStructured_impl::dimen ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured.dimen)
  return this->vdimen;
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured.dimen)
}

/**
 * Method:  lower[]
 */
int32_t
TOPS::MatrixStructured_impl::lower (
  /* in */ int32_t a ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured.lower)
  return this->vlower[a];
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured.lower)
}

/**
 * Method:  length[]
 */
int32_t
TOPS::MatrixStructured_impl::length (
  /* in */ int32_t a ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured.length)
  return this->vlength[a];
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured.length)
}

/**
 * Method:  set[D1]
 */
void
TOPS::MatrixStructured_impl::set (
  /* in */ int32_t i,
  /* in */ ::sidl::array<double> values ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured.setD1)
  // Insert-Code-Here {TOPS.MatrixStructured.setD1} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured.setD1)
}

/**
 * Method:  set[D2]
 */
void
TOPS::MatrixStructured_impl::set (
  /* in */ int32_t i,
  /* in */ int32_t j,
  /* in */ ::sidl::array<double> values ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured.setD2)
  int I = i - this->gghostlower[0] + this->gghostlength[0]*(j - this->gghostlower[1]);
  if ((values.dimen() == 1 || values.length(1) == 1) && values.length(0) == 1) {
    MatSetValuesLocal(this->mat,1,&I,1,&I,values.first(),INSERT_VALUES);
  } else {
    MatSetValuesRowLocal(this->mat,I,values.first());
  }
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured.setD2)
}

/**
 * Method:  set[D3]
 */
void
TOPS::MatrixStructured_impl::set (
  /* in */ int32_t i,
  /* in */ int32_t j,
  /* in */ int32_t k,
  /* in */ ::sidl::array<double> values ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured.setD3)
  int I = i - this->gghostlower[0] + this->gghostlength[0]*(j - this->gghostlower[1]) + this->gghostlength[0]*this->gghostlength[1]*(k - this->gghostlower[2]);
  if ((values.dimen() == 1 || values.length(1) == 1) && values.length(0) == 1) {
    MatSetValuesLocal(this->mat,1,&I,1,&I,values.first(),INSERT_VALUES);
  } else {
    MatSetValuesRowLocal(this->mat,I,values.first());
  }
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured.setD3)
}

/**
 * Method:  set[D4]
 */
void
TOPS::MatrixStructured_impl::set (
  /* in */ int32_t i,
  /* in */ int32_t j,
  /* in */ int32_t k,
  /* in */ int32_t l,
  /* in */ ::sidl::array<double> values ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured.setD4)
  int I = i - this->gghostlower[0] + this->gghostlength[0]*(j - this->gghostlower[1]) + this->gghostlength[0]*this->gghostlength[1]*(k - this->gghostlower[2]) +
          this->gghostlength[0]*this->gghostlength[1]*this->gghostlength[2]*(l - gghostlower[3]);
  if ((values.dimen() == 1 || values.length(1) == 1) && values.length(0) == 1) {
    MatSetValuesLocal(this->mat,1,&I,1,&I,values.first(),INSERT_VALUES);
  } else {
    MatSetValuesRowLocal(this->mat,I,values.first());
  }
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured.setD4)
}

/**
 * Method:  zero[]
 */
void
TOPS::MatrixStructured_impl::zero ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured.zero)
  MatZeroEntries(this->mat);
  // DO-NOT-DELETE splicer.end(TOPS.MatrixStructured.zero)
}


// DO-NOT-DELETE splicer.begin(TOPS.MatrixStructured._misc)
// Insert-Code-Here {TOPS.MatrixStructured._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.MatrixStructured._misc)

