#ifndef __PETSc_Map_h__
#define __PETSc_Map_h__

// this contains the PETSc definition of (Petsc)Map
#include "petscvec.h"

#include "esi/petsc/object.h"

// The PETSc_Map supports the esi::Map and esi::MapPartition interfaces

#include "esi/Map.h"
#include "esi/MapPartition.h"

namespace esi{namespace petsc{

/**=========================================================================**/
template<class Ordinal> 
class Map : public virtual esi::MapPartition<Ordinal>, public virtual esi::petsc::Object
{
  public:

    // constructor.
    Map(MPI_Comm comm) {};	

    // Construct a map from a map.
    Map(esi::Map<Ordinal>& sourceMap);

    // Construct a map from a PETSc (old-style) map.
    Map(PetscMap sourceMap);

    // Basic constructor
    Map(MPI_Comm comm, int n, int N);

    // destructor.
    virtual ~Map();

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface);
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list){return 1;};


    //  Interface for esi::Map  ---------------

    // Get the size of this mapped dimension of the problem.
    virtual esi::ErrorCode getGlobalSize(Ordinal& globalSize);
    virtual esi::ErrorCode getLocalSize(Ordinal& localSize);

    //  Interface for esi::MapPartition  ---------------

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




