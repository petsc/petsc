#ifndef __PETSc_Object_h__
#define __PETSc_Object_h__

#include "petsc.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

// #include "esi/ESI.h"
#define ESI_NO_COMPLEX
#include "esi/basicTypes.h"
#include "esi/ordinalTraits.h"
#include "esi/scalarTraits.h"
#include "esi/Argv.h"
#include "esi/Object.h"
#include "esi/MPI_traits.h"

namespace esi{namespace petsc{

/**=========================================================================**/
class Object : public virtual esi::Object
{
  public:

    // Constructor.
    Object()               {comm = 0; refcnt = 1;};	

    // Destructor.
    virtual ~Object(void) {};


    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode setRunTimeModel(const char * name,void * comm);
    virtual esi::ErrorCode getRunTimeModel(const char * name,void * & comm);
    virtual esi::ErrorCode getRunTimeModelsSupported(esi::Argv * list);

    virtual esi::ErrorCode getInterface(const char* name, void*& iface);
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);

    virtual esi::ErrorCode addReference() ;
    virtual esi::ErrorCode deleteReference() ;


  protected:
    MPI_Comm    comm;
    int         refcnt;
};

}}

extern int ESILoadFactory(char *commname,void *comm,char *classname,void *&f);

#endif




