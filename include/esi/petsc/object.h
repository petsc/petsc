#ifndef __PETSc_Object_h__
#define __PETSc_Object_h__

#include "petsc.h"

//#include "esi/ESI.h"
#include "esi/basicTypes.h"
#include "esi/ordinalTraits.h"
#include "esi/scalarTraits.h"
#include "esi/Argv.h"
#include "esi/Object.h"

#include "ESI-MPI.h"


#include "esi/config.h"
#include "esi/MPI_traits.h"

namespace esi {
  typedef esi_int  ErrorCode;
  typedef esi_msg  ErrorMsg;
};

namespace esi{namespace petsc{

/**=========================================================================**/
class Object : public virtual esi::Object
{
  public:

    // Constructors.
    Object(MPI_Comm icomm) {comm = icomm;};	
    Object()               {comm = 0; pobject = 0; refcnt = 1;};	

    // Destructor.
    virtual ~Object() {};


    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode setRunTimeModel(const char * name,void * comm);
    virtual esi::ErrorCode getRunTimeModel(const char * name,void * & comm);
    virtual esi::ErrorCode getRunTimeModelsSupported(esi::Argv * list);

    virtual esi::ErrorCode getInterface(const char* name, void*& iface);
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);

    virtual void addReference() ;
    virtual void deleteReference() ;


  protected:
    MPI_Comm    comm;
    PetscObject pobject;
    int         refcnt;
};

}}

#endif




