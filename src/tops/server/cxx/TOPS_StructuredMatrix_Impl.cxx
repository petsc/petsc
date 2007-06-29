// 
// File:          TOPS_StructuredMatrix_Impl.cxx
// Symbol:        TOPS.StructuredMatrix-v0.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Server-side implementation for TOPS.StructuredMatrix
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "TOPS_StructuredMatrix_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._includes)
// Insert-Code-Here {TOPS.StructuredMatrix._includes} (additional includes or code)
#include "petscmat.h"
// DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._includes)

// speical constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
TOPS::StructuredMatrix_impl::StructuredMatrix_impl() : StubBase(
  reinterpret_cast< void*>(::TOPS::StructuredMatrix::_wrapObj(reinterpret_cast< 
  void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._ctor2)
  // Insert-Code-Here {TOPS.StructuredMatrix._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._ctor2)
}

// user defined constructor
void TOPS::StructuredMatrix_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._ctor)
  // Insert-Code-Here {TOPS.StructuredMatrix._ctor} (constructor)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._ctor)
}

// user defined destructor
void TOPS::StructuredMatrix_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._dtor)
  // Insert-Code-Here {TOPS.StructuredMatrix._dtor} (destructor)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._dtor)
}

// static class initializer
void TOPS::StructuredMatrix_impl::_load() {
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._load)
  // Insert-Code-Here {TOPS.StructuredMatrix._load} (class initialization)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  getDimen[]
 */
int32_t
TOPS::StructuredMatrix_impl::getDimen_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.getDimen)
  // Insert-Code-Here {TOPS.StructuredMatrix.getDimen} (getDimen method)
  
  return this->vdimen;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.getDimen)
}

/**
 * Method:  getLower[]
 */
int32_t
TOPS::StructuredMatrix_impl::getLower_impl (
  /* in */int32_t dimen ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.getLower)
  // Insert-Code-Here {TOPS.StructuredMatrix.getLower} (getLower method)

  return this->vlower[dimen];
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.getLower)
}

/**
 * Method:  getLength[]
 */
int32_t
TOPS::StructuredMatrix_impl::getLength_impl (
  /* in */int32_t dimen ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.getLength)
  // Insert-Code-Here {TOPS.StructuredMatrix.getLength} (getLength method)

  return this->vlength[dimen];
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.getLength)
}

/**
 * Method:  setDimen[]
 */
void
TOPS::StructuredMatrix_impl::setDimen_impl (
  /* in */int32_t dim ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setDimen)
  // Insert-Code-Here {TOPS.StructuredMatrix.setDimen} (setDimen method)
  this->vdimen = dim;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setDimen)
}

/**
 * Method:  setLower[]
 */
void
TOPS::StructuredMatrix_impl::setLower_impl (
  /* in array<int,3> */::sidl::array<int32_t> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setLower)
  // Insert-Code-Here {TOPS.StructuredMatrix.setLower} (setLower method)
  for (int i = 0; i < this->vdimen; ++i) 
  	this->vlower[i] = values.get(i);
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setLower)
}

/**
 * Method:  setLength[]
 */
void
TOPS::StructuredMatrix_impl::setLength_impl (
  /* in array<int,3> */::sidl::array<int32_t> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setLength)
  // Insert-Code-Here {TOPS.StructuredMatrix.setLength} (setLength method)
  for (int i = 0; i < this->vdimen; ++i) 
  	this->vlength[i] = values.get(i); 
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setLength)
}

/**
 * Method:  setGhostLower[]
 */
void
TOPS::StructuredMatrix_impl::setGhostLower_impl (
  /* in array<int,3> */::sidl::array<int32_t> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setGhostLower)
  // Insert-Code-Here {TOPS.StructuredMatrix.setGhostLower} (setGhostLower method)
  for (int i = 0; i < this->vdimen; ++i) 
  	this->gghostlower[i] = values.get(i); 
 
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setGhostLower)
}

/**
 * Method:  setGhostLength[]
 */
void
TOPS::StructuredMatrix_impl::setGhostLength_impl (
  /* in array<int,3> */::sidl::array<int32_t> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setGhostLength)
  // Insert-Code-Here {TOPS.StructuredMatrix.setGhostLength} (setGhostLength method)
  for (int i = 0; i < this->vdimen; ++i) 
  	this->gghostlength[i] = values.get(i); 
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setGhostLength)
}

/**
 * Method:  setMat[]
 */
void
TOPS::StructuredMatrix_impl::setMat_impl (
  /* in */void* m ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setMat)
  // Insert-Code-Here {TOPS.StructuredMatrix.setMat} (setMat method)
  this->mat = (Mat)m;
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setMat)
}

/**
 * Method:  set[D1]
 */
void
TOPS::StructuredMatrix_impl::set_impl (
  /* in */int32_t i,
  /* in array<double,2> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setD1)
  // Insert-Code-Here {TOPS.StructuredMatrix.setD1} (set method)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setD1)
}

/**
 * Method:  set[D2]
 */
void
TOPS::StructuredMatrix_impl::set_impl (
  /* in */int32_t i,
  /* in */int32_t j,
  /* in array<double,2> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setD2)
  int I = i - this->gghostlower[0] + this->gghostlength[0]*(j - this->gghostlower[1]);
  if ((values.dimen() == 1 || values.length(1) == 1) && values.length(0) == 1) {
    MatSetValuesLocal(this->mat,1,&I,1,&I,values.first(),INSERT_VALUES);
  } else {
    MatSetValuesRowLocal(this->mat,I,values.first());
  }
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setD2)
}

/**
 * Method:  set[D3]
 */
void
TOPS::StructuredMatrix_impl::set_impl (
  /* in */int32_t i,
  /* in */int32_t j,
  /* in */int32_t k,
  /* in array<double,2> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setD3)
  int I = i - this->gghostlower[0] + this->gghostlength[0]*(j - this->gghostlower[1]) + this->gghostlength[0]*this->gghostlength[1]*(k - this->gghostlower[2]);
  if ((values.dimen() == 1 || values.length(1) == 1) && values.length(0) == 1) {
    MatSetValuesLocal(this->mat,1,&I,1,&I,values.first(),INSERT_VALUES);
  } else {
    MatSetValuesRowLocal(this->mat,I,values.first());
  }
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setD3)
}

/**
 * Method:  set[D4]
 */
void
TOPS::StructuredMatrix_impl::set_impl (
  /* in */int32_t i,
  /* in */int32_t j,
  /* in */int32_t k,
  /* in */int32_t l,
  /* in array<double,2> */::sidl::array<double> values ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setD4)
  int I = i - this->gghostlower[0] + this->gghostlength[0]*(j - this->gghostlower[1]) + this->gghostlength[0]*this->gghostlength[1]*(k - this->gghostlower[2]) +
          this->gghostlength[0]*this->gghostlength[1]*this->gghostlength[2]*(l - gghostlower[3]);
  if ((values.dimen() == 1 || values.length(1) == 1) && values.length(0) == 1) {
    MatSetValuesLocal(this->mat,1,&I,1,&I,values.first(),INSERT_VALUES);
  } else {
    MatSetValuesRowLocal(this->mat,I,values.first());
  }
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setD4)
}

/**
 * Method:  apply[]
 */
void
TOPS::StructuredMatrix_impl::apply_impl (
  /* in array<double> */::sidl::array<double> x,
  /* in array<double> */::sidl::array<double> y ) 
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.apply)
  // Insert-Code-Here {TOPS.StructuredMatrix.apply} (apply method)
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.apply)
}

/**
 * Method:  zero[]
 */
void
TOPS::StructuredMatrix_impl::zero_impl () 

{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.zero)
  MatZeroEntries(this->mat);
  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.zero)
}

/**
 *  Starts up a component presence in the calling framework.
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
TOPS::StructuredMatrix_impl::setServices_impl (
  /* in */::gov::cca::Services services ) 
// throws:
//     ::gov::cca::CCAException
//     ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix.setServices)
  // Insert-Code-Here {TOPS.StructuredMatrix.setServices} (setServices method)
#undef __FUNCT__
#define __FUNCT__ "TOPS::StructuredMatrix::setServices"

  myServices = services;

  gov::cca::Port p = (*this);      //  Babel required casting
  if(p._is_nil()) {
    fprintf(stderr, "Error:: %s:%d: Error casting (*this) to gov::cca::Port \n",
	    __FILE__, __LINE__);
    return;
  }
  
  // Provides ports
  myServices.addProvidesPort(p,
			   "TOPS.Matrix",
			   "TOPS.Matrix", myServices.createTypeMap());

  myServices.addProvidesPort(p,
			   "TOPS.Structured.Matrix",
			   "TOPS.Structured.Matrix", myServices.createTypeMap());

  // DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix.setServices)
}


// DO-NOT-DELETE splicer.begin(TOPS.StructuredMatrix._misc)
// Insert-Code-Here {TOPS.StructuredMatrix._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(TOPS.StructuredMatrix._misc)

