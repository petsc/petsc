// 
// File:          TOPS_Unstructured_Matrix_Impl.cc
// Symbol:        TOPS.Unstructured.Matrix-v0.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Server-side implementation for TOPS.Unstructured.Matrix
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.10.12
// 
#include "TOPS_Unstructured_Matrix_Impl.hh"

// DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._includes)
// Insert-Code-Here {TOPS.Unstructured.Matrix._includes} (additional includes or code)
// DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._includes)

// user-defined constructor.
void TOPS::Unstructured::Matrix_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._ctor)
  // Insert-Code-Here {TOPS.Unstructured.Matrix._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._ctor)
}

// user-defined destructor.
void TOPS::Unstructured::Matrix_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._dtor)
  // Insert-Code-Here {TOPS.Unstructured.Matrix._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._dtor)
}

// static class initializer.
void TOPS::Unstructured::Matrix_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._load)
  // Insert-Code-Here {TOPS.Unstructured.Matrix._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._load)
}

// user-defined static methods: (none)

// user-defined non-static methods:
/**
 * Method:  set[Point]
 */
void
TOPS::Unstructured::Matrix_impl::set (
  /* in */ int32_t row,
  /* in */ int32_t column,
  /* in */ ::sidl::array<double> values ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.setPoint)
  // Insert-Code-Here {TOPS.Unstructured.Matrix.setPoint} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.setPoint)
}

/**
 * Method:  set[Row]
 */
void
TOPS::Unstructured::Matrix_impl::set (
  /* in */ int32_t row,
  /* in */ ::sidl::array<int32_t> columns,
  /* in */ ::sidl::array<double> values ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.setRow)
  // Insert-Code-Here {TOPS.Unstructured.Matrix.setRow} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.setRow)
}

/**
 * Method:  set[Column]
 */
void
TOPS::Unstructured::Matrix_impl::set (
  /* in */ ::sidl::array<int32_t> rows,
  /* in */ int32_t column,
  /* in */ ::sidl::array<double> values ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.setColumn)
  // Insert-Code-Here {TOPS.Unstructured.Matrix.setColumn} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.setColumn)
}

/**
 * Method:  set[]
 */
void
TOPS::Unstructured::Matrix_impl::set (
  /* in */ ::sidl::array<int32_t> rows,
  /* in */ ::sidl::array<int32_t> columns,
  /* in */ ::sidl::array<double> values ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.set)
  MatSetValuesLocal(this->mat,rows.length(0),rows.first(),columns.length(0),columns.first(),values.first(),ADD_VALUES);
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.set)
}

/**
 * Method:  apply[]
 */
void
TOPS::Unstructured::Matrix_impl::apply (
  /* in */ ::sidl::array<double> x,
  /* in */ ::sidl::array<double> y ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.apply)
  // Insert-Code-Here {TOPS.Unstructured.Matrix.apply} (apply method)
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.apply)
}

/**
 * Method:  zero[]
 */
void
TOPS::Unstructured::Matrix_impl::zero ()
throw () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix.zero)
  MatZeroEntries(this->mat);
  // DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix.zero)
}


// DO-NOT-DELETE splicer.begin(TOPS.Unstructured.Matrix._misc)
// Insert-Code-Here {TOPS.Unstructured.Matrix._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.Unstructured.Matrix._misc)

