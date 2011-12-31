// 
// File:          TOPS_Structured_Matrix_Impl.cxx
// Symbol:        TOPS.Structured.Matrix-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for TOPS.Structured.Matrix
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "TOPS_Structured_Matrix_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._includes)
// Insert-Code-Here {TOPS.Structured.Matrix._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
TOPS::Structured::Matrix_impl::Matrix_impl() : StubBase(reinterpret_cast< 
  void*>(::TOPS::Structured::Matrix::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._ctor2)
  // Insert-Code-Here {TOPS.Structured.Matrix._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._ctor2)
}

// user defined constructor
void TOPS::Structured::Matrix_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._ctor)
  // Insert-Code-Here {TOPS.Structured.Matrix._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._ctor)
}

// user defined destructor
void TOPS::Structured::Matrix_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._dtor)
  // Insert-Code-Here {TOPS.Structured.Matrix._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._dtor)
}

// static class initializer
void TOPS::Structured::Matrix_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._load)
  // Insert-Code-Here {TOPS.Structured.Matrix._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  dimen[]
 */
int32_t
TOPS::Structured::Matrix_impl::dimen_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix.dimen)
  return this->vdimen;
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix.dimen)
}

/**
 * Method:  lower[]
 */
int32_t
TOPS::Structured::Matrix_impl::lower_impl (
  /* in */int32_t a ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix.lower)
  return this->vlower[a];
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix.lower)
}

/**
 * Method:  length[]
 */
int32_t
TOPS::Structured::Matrix_impl::length_impl (
  /* in */int32_t a ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix.length)
  return this->vlength[a];
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix.length)
}

/**
 * Method:  set[D1]
 */
void
TOPS::Structured::Matrix_impl::set_impl (
  /* in */int32_t i,
  /* in array<double,2> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix.setD1)
  // Insert-Code-Here {TOPS.Structured.Matrix.setD1} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix.setD1)
}

/**
 * Method:  set[D2]
 */
void
TOPS::Structured::Matrix_impl::set_impl (
  /* in */int32_t i,
  /* in */int32_t j,
  /* in array<double,2> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix.setD2)
  int I = i - this->gghostlower[0] + this->gghostlength[0]*(j - this->gghostlower[1]);
  if ((values.dimen() == 1 || values.length(1) == 1) && values.length(0) == 1) {
    MatSetValuesLocal(this->mat,1,&I,1,&I,values.first(),INSERT_VALUES);
  } else {
    MatSetValuesRowLocal(this->mat,I,values.first());
  }
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix.setD2)
}

/**
 * Method:  set[D3]
 */
void
TOPS::Structured::Matrix_impl::set_impl (
  /* in */int32_t i,
  /* in */int32_t j,
  /* in */int32_t k,
  /* in array<double,2> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix.setD3)
  int I = i - this->gghostlower[0] + this->gghostlength[0]*(j - this->gghostlower[1]) + this->gghostlength[0]*this->gghostlength[1]*(k - this->gghostlower[2]);
  if ((values.dimen() == 1 || values.length(1) == 1) && values.length(0) == 1) {
    MatSetValuesLocal(this->mat,1,&I,1,&I,values.first(),INSERT_VALUES);
  } else {
    MatSetValuesRowLocal(this->mat,I,values.first());
  }
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix.setD3)
}

/**
 * Method:  set[D4]
 */
void
TOPS::Structured::Matrix_impl::set_impl (
  /* in */int32_t i,
  /* in */int32_t j,
  /* in */int32_t k,
  /* in */int32_t l,
  /* in array<double,2> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix.setD4)
  int I = i - this->gghostlower[0] + this->gghostlength[0]*(j - this->gghostlower[1]) + this->gghostlength[0]*this->gghostlength[1]*(k - this->gghostlower[2]) +
          this->gghostlength[0]*this->gghostlength[1]*this->gghostlength[2]*(l - gghostlower[3]);
  if ((values.dimen() == 1 || values.length(1) == 1) && values.length(0) == 1) {
    MatSetValuesLocal(this->mat,1,&I,1,&I,values.first(),INSERT_VALUES);
  } else {
    MatSetValuesRowLocal(this->mat,I,values.first());
  }
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix.setD4)
}

/**
 * Method:  apply[]
 */
void
TOPS::Structured::Matrix_impl::apply_impl (
  /* in array<double> */::sidl::array<double> x,
  /* in array<double> */::sidl::array<double> y ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix.apply)
  // Insert-Code-Here {TOPS.Structured.Matrix.apply} (apply method)
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix.apply)
}

/**
 * Method:  zero[]
 */
void
TOPS::Structured::Matrix_impl::zero_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix.zero)
  MatZeroEntries(this->mat);
  // DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix.zero)
}


// DO-NOT-DELETE splicer.begin(TOPS.Structured.Matrix._misc)
// Insert-Code-Here {TOPS.Structured.Matrix._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.Structured.Matrix._misc)

