#ifndef __PETSc_IndexSpace_h__
#define __PETSc_IndexSpace_h__

// this contains the definition of PetscMap
#include "petscvec.h"

#include "esi/petsc/object.h"

// The esi::petsc::IndexSpace supports the esi::IndexSpace interfaces

#include "esi/IndexSpace.h"

namespace esi{namespace petsc{

/**=========================================================================**/
template<class Ordinal> 
class IndexSpace : public virtual esi::IndexSpace<Ordinal>, public virtual esi::petsc::Object
{
  public:

    // constructor.
    IndexSpace(MPI_Comm comm) {};	

    // Construct a map from a map.
    IndexSpace(esi::IndexSpace<Ordinal>& sourceIndexSpace);

    // Construct a map from a PETSc (old-style) map.
    IndexSpace(PetscMap sourceIndexSpace);

    // Basic constructor
    IndexSpace(MPI_Comm comm, int n, int N);

    // destructor.
    virtual ~IndexSpace();

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface);
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list){return 1;};


    //  Interface for esi::IndexSpace  ---------------

    // Get the size of this mapped dimension of the problem.
    virtual esi::ErrorCode getGlobalSize(Ordinal& globalSize);
    virtual esi::ErrorCode getLocalSize(Ordinal& localSize);

    // Get the size of this dimension of the problem, as well as 
    // the global offset info for all processors.
    virtual esi::ErrorCode getGlobalPartitionSizes(Ordinal* globalSizes);
    virtual esi::ErrorCode getGlobalPartitionOffsets(Ordinal* globalOffsets);

    virtual esi::ErrorCode getGlobalPartitionSetSize(Ordinal &) {return 1;};
    virtual esi::ErrorCode getLocalPartitionRank(Ordinal &) {return 1;};

    virtual esi::ErrorCode getGlobalColorSetSize(Ordinal &) {return 1;};
    virtual esi::ErrorCode getLocalColors(Ordinal *) {return 1;};
    virtual esi::ErrorCode getLocalIdentifiers(Ordinal *) {return 1;};

    // Get the local size offset info in this dimension.
    virtual esi::ErrorCode getLocalPartitionOffset(Ordinal& localOffset);

  private:
    PetscMap map;
};
}}

#endif




