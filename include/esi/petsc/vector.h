#ifndef __PETSc_Vector_h__
#define __PETSc_Vector_h__

// The esi::petsc::Vector supports the 
//    esi::Vector<Scalar,Ordinal>
//    esi::Vector<Scalar,Ordinal>ReplaceAccess interfaces

#include "esi/petsc/indexspace.h"

#include "esi/Vector.h"
#include "esi/VectorReplaceAccess.h"

// this contains the PETSc definition of Vector
#include "petscvec.h"

namespace esi{namespace petsc{

/**=========================================================================**/
template<class Scalar,class Ordinal>
  class Vector : public virtual esi::VectorReplaceAccess<Scalar,Ordinal>, public esi::petsc::Object
{
  public:

    // Destructor.
    virtual ~Vector(void);

    // Construct a Vector from a IndexSpace.
    Vector(  esi::IndexSpace<Ordinal> *source);

    // Construct a Vector from a PETSc Vector
    Vector(Vec pvec);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface);
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);


    //  Interface for ESI_Vector  ---------------
    
    virtual esi::ErrorCode clone(esi::Vector<Scalar,Ordinal>*& x);
    virtual esi::ErrorCode getGlobalSize( Ordinal & dim) ;
    virtual esi::ErrorCode getLocalSize( Ordinal & dim) ;
    virtual esi::ErrorCode getIndexSpace(  esi::IndexSpace<Ordinal>*& outmap)  ;
    virtual esi::ErrorCode copy( esi::Vector<Scalar,Ordinal>& x) ;   
    virtual esi::ErrorCode put(  Scalar scalar) ;
    virtual esi::ErrorCode scale(  Scalar scalar) ;
    virtual esi::ErrorCode scaleDiagonal(  esi::Vector<Scalar,Ordinal>& x) ;
    virtual esi::ErrorCode norm1( Scalar& norm)   ;
    virtual esi::ErrorCode norm2( Scalar& norm)   ;
    virtual esi::ErrorCode norm2squared( Scalar& norm)   ;
    virtual esi::ErrorCode normInfinity( Scalar& norm)   ;
    virtual esi::ErrorCode dot( esi::Vector<Scalar,Ordinal>& x, Scalar& product)   ;
    virtual esi::ErrorCode axpy( esi::Vector<Scalar,Ordinal>& x, Scalar scalar) ;
    virtual esi::ErrorCode aypx(Scalar scalar, esi::Vector<Scalar,Ordinal>& x) ;

    virtual esi::ErrorCode minAbsCoef(Scalar &)  {return 1;}
    virtual esi::ErrorCode axpby(Scalar,esi::Vector<Scalar,Ordinal>&,Scalar,esi::Vector<Scalar,Ordinal>&);
    virtual esi::ErrorCode getCoefPtrReadLock(Scalar *&) ;
    virtual esi::ErrorCode getCoefPtrReadWriteLock(Scalar *&);
    virtual esi::ErrorCode releaseCoefPtrLock(Scalar *&) ;

    // Interface for ESI_VectorReplaceAccess
   
    virtual esi::ErrorCode setArrayPointer(Scalar* array, Ordinal length);

  private:
    Vec                      vec;
    esi::IndexSpace<Ordinal> *map;
};

}
  /* -------------------------------------------------------------------------*/

template<class Scalar,class Ordinal> class VectorFactory 
#if defined(PETSC_HAVE_CCA)
           :  public virtual gov::cca::Port, public virtual gov::cca::Component
#endif
{
  public:

    // Destructor.
  virtual ~VectorFactory(void){};

    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *) = 0;
#endif

    // Construct a Vector
    virtual esi::ErrorCode getVector(esi::IndexSpace<Ordinal>&,esi::Vector<Scalar,Ordinal>*&v) = 0; 
};

}
EXTERN int VecESIWrap(Vec,esi::Vector<double,int>**);

#endif




