#ifndef __PETSc_Vector_h__
#define __PETSc_Vector_h__

// The PETSc_Vector supports the 
//    esi::Vector<Scalar,Ordinal>
//    esi::Vector<Scalar,Ordinal>ReplaceAccess interfaces

#include "esi/petsc/map.h"

#include "esi/Vector.h"
#include "esi/VectorReplaceAccess.h"

// this contains the PETSc definition of Vector
#include "petscvec.h"

namespace esi{namespace petsc{

/**=========================================================================**/
template<class Scalar,class Ordinal>
  class Vector : public virtual esi::VectorReplaceAccess<Scalar,Ordinal>, public virtual esi::petsc::Object
{
  public:

    // Destructor.
    virtual ~Vector();

    // Construct a Vector from a Map.
    Vector(  esi::MapPartition<Ordinal> *source);

    // Construct a Vector from a PETSc Vector
    Vector(Vec pvec);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface);
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);


    //  Interface for ESI_Vector  ---------------
    
    virtual esi::ErrorCode clone(esi::Vector<Scalar,Ordinal>*& x);
    virtual esi::ErrorCode getGlobalSize( Ordinal & dim) ;
    virtual esi::ErrorCode getLocalSize( Ordinal & dim) ;
    virtual esi::ErrorCode getMapPartition(  esi::MapPartition<Ordinal>*& outmap)  ;
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
    virtual esi::ErrorCode axpby(Scalar, esi::Vector<Scalar,Ordinal> &, Scalar,esi::Vector<Scalar,Ordinal> &);
    virtual esi::ErrorCode getCoefPtrReadLock(Scalar *&) ;
    virtual esi::ErrorCode getCoefPtrReadWriteLock(Scalar *&);
    virtual esi::ErrorCode releaseCoefPtrLock(Scalar *&) ;

    // Interface for ESI_VectorReplaceAccess
   
    virtual esi::ErrorCode setArrayPointer(Scalar* array, Ordinal length);

    // Obtain access to ACTUAL PETSc vector
    // Should be private somehow

    virtual esi::ErrorCode getPETScVec(Vec *);

  private:
    Vec                        vec;
    esi::MapPartition<Ordinal> *map;

};

}}

#endif




