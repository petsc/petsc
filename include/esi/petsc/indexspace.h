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
class IndexSpace : public virtual esi::IndexSpace<Ordinal>, public esi::petsc::Object
{
  public:

    // constructor.
    IndexSpace(MPI_Comm icomm) {};	

    // Construct an IndexSpace form an IndexSpace 
    IndexSpace(esi::IndexSpace<Ordinal>& sourceIndexSpace);

    // Construct an IndexSpace from a PETSc (old-style) map.
    IndexSpace(PetscMap sourceIndexSpace);

    // Basic constructor
    IndexSpace(MPI_Comm comm, int n, int N);

    // destructor.
    virtual ~IndexSpace(void);

    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode getInterface(const char* name, void*& iface);
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);

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
}

  /* -------------------------------------------------------------------------*/

template<class Ordinal> class IndexSpaceFactory 
#if defined(PETSC_HAVE_CCA)
           :  public virtual gov::cca::Port, public virtual gov::cca::Component
#endif
{
  public:

    // Destructor.
  virtual ~IndexSpaceFactory(void){};

    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *) = 0;
#endif

    // Construct a IndexSpace
    virtual esi::ErrorCode getIndexSpace(const char * name,void *comm,int m,esi::IndexSpace<Ordinal>*&v) = 0; 
};
}

extern int ESICreateIndexSpace(const char * commname,void *comm,int m,::esi::IndexSpace<int>*&v);
#endif




