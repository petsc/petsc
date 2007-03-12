// 
// File:          TOPS_Unstructured_Matrix_Impl.cxx
// Symbol:        TOPS.Unstructured.Matrix-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for TOPS.Unstructured.Matrix
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "TOPS_Unstructured_Matrix_Impl.hxx"

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
// DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._includes)
// Insert-Code-Here {TOPS.Unstructured.Matrix._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
TOPS::Unstructured::Matrix_impl::Matrix_impl() : StubBase(reinterpret_cast< 
  void*>(::TOPS::Unstructured::Matrix::_wrapObj(reinterpret_cast< void*>(
  this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._ctor2)
  // Insert-Code-Here {TOPS.Unstructured.Matrix._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._ctor2)
}

// user defined constructor
void TOPS::Unstructured::Matrix_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._ctor)
  // Insert-Code-Here {TOPS.Unstructured.Matrix._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._ctor)
}

// user defined destructor
void TOPS::Unstructured::Matrix_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._dtor)
  // Insert-Code-Here {TOPS.Unstructured.Matrix._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._dtor)
}

// static class initializer
void TOPS::Unstructured::Matrix_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._load)
  // Insert-Code-Here {TOPS.Unstructured.Matrix._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  set[Point]
 */
void
TOPS::Unstructured::Matrix_impl::set_impl (
  /* in */int32_t row,
  /* in */int32_t column,
  /* in array<double> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.setPoint)
  // Insert-Code-Here {TOPS.Unstructured.Matrix.setPoint} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.setPoint)
}

/**
 * Method:  set[Row]
 */
void
TOPS::Unstructured::Matrix_impl::set_impl (
  /* in */int32_t row,
  /* in array<int> */::sidl::array<int32_t> columns,
  /* in array<double> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.setRow)
  // Insert-Code-Here {TOPS.Unstructured.Matrix.setRow} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.setRow)
}

/**
 * Method:  set[Column]
 */
void
TOPS::Unstructured::Matrix_impl::set_impl (
  /* in array<int> */::sidl::array<int32_t> rows,
  /* in */int32_t column,
  /* in array<double> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.setColumn)
  // Insert-Code-Here {TOPS.Unstructured.Matrix.setColumn} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.setColumn)
}

/**
 * Method:  set[]
 */
void
TOPS::Unstructured::Matrix_impl::set_impl (
  /* in array<int> */::sidl::array<int32_t> rows,
  /* in array<int> */::sidl::array<int32_t> columns,
  /* in array<double> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.set)
  MatSetValuesLocal(this->mat,rows.length(0),rows.first(),columns.length(0),columns.first(),values.first(),ADD_VALUES);
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.set)
}

/**
 * Method:  apply[]
 */
void
TOPS::Unstructured::Matrix_impl::apply_impl (
  /* in array<double> */::sidl::array<double> x,
  /* in array<double> */::sidl::array<double> y ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.apply)
  // Insert-Code-Here {TOPS.Unstructured.Matrix.apply} (apply method)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.apply)
}

/**
 * Method:  zero[]
 */
void
TOPS::Unstructured::Matrix_impl::zero_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.zero)
  MatZeroEntries(this->mat);
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.zero)
}


// DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._misc)
// Insert-Code-Here {TOPS.Unstructured.Matrix._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._misc)

