#ifndef __PETSc_Object_h__
#define __PETSc_Object_h__

#include "petsc.h"

//#include "esi/ESI.h"
#include "esi/basicTypes.h"
#include "esi/ordinalTraits.h"
#include "esi/scalarTraits.h"
#include "esi/Argv.h"
#include "esi/Object.h"
#include "esi/ESI-MPI.h"
#include "esi/config.h"
#include "esi/MPI_traits.h"

#if defined(PETSC_HAVE_CCA)
#include "cca.h"
#endif

namespace esi{namespace petsc{

/**=========================================================================**/
class Object : public virtual esi::Object
#if defined(PETSC_HAVE_CCA)
             , public virtual gov::cca::Port, public virtual gov::cca::Component
#endif
{
  public:

    // Constructor.
    Object()               {comm = 0; pobject = 0; refcnt = 1;};	

    // Destructor.
    virtual ~Object() {};


    //  Interface for esi::Object  ---------------

    virtual esi::ErrorCode setRunTimeModel(const char * name,void * comm);
    virtual esi::ErrorCode getRunTimeModel(const char * name,void * & comm);
    virtual esi::ErrorCode getRunTimeModelsSupported(esi::Argv * list);

    virtual esi::ErrorCode getInterface(const char* name, void*& iface);
    virtual esi::ErrorCode getInterfacesSupported(esi::Argv * list);

    virtual esi::ErrorCode addReference() ;
    virtual esi::ErrorCode deleteReference() ;


    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *);
#endif

  protected:
    MPI_Comm    comm;
    PetscObject pobject;
    int         refcnt;
};

}}

#if defined(PETSC_HAVE_CCA)
namespace gov{namespace cca{namespace petsc{
class Services  {
  public:
    /** obligatory virtual destructor */
    virtual ~Services (){}

    /** Creates a PortInfo to be used in subsequent
          calls to describe a Port. The arguments given are copied, not kept.
          properties is an even length list (argv) of key/value pairs 
          terminated by a NULL key. Note that names of Ports in a 
          component are not distinct for used and provided ports. 
          Standard properties: (UNADOPTED, extensible list)
                gov.cca.port.minConnections (int >= 0; default 0)
		gov.cca.port.maxConnections (int >= 1, default 1)
		gov.cca.port.proxyAllowed   (true,false; default false)
      */
    virtual PortInfo *  createPortInfo(CONST char *name, CONST char *type, CONST char** properties){};

      /**  Fetch a port from the framework. Normally this means a uses port.
      If no uses port is connected and a provided port of the name requested
      is available, the component will be handed back that port.
       Returns NULL if no provided port or uses port connection matches name.
       @see Services.java; UNADOPTED C++ definition of "an error occurs".
   */
    virtual Port *getPort(CONST char *name){};

      /** Free's the port indicated by the instance name for modification
	  by the component's containing framework.  After this call the
	  port will not be valid for use until getPort() is called
	  again.*/
    virtual void releasePort(CONST char *name){};

      /** Notifies the framework that a port described by PortInfo
	  may be used by this component. The portinfo is obtained 
      from createPortInfo. Returns nonzero if there is an error
      in registering, such as bad PortInfo or already registered.
       @see Services.java; UNADOPTED C++ definition of "an error occurs".
      */
    virtual int registerUsesPort(PortInfo *name_and_type){};

      /** Notify the framework that a Port, previously
          registered by this component, is no longer desired.
          Returns nonzero if the port is still in use, ignoring
          the unregister request.
       @see Services.java; UNADOPTED C++ definition of "an error occurs".
       */
    virtual int unregisterUsesPort(CONST char *name){};

      /** Exports a Port implemented by this component to the framework.  
	  This Port is now available for the framework to
	  connect to other components. The PortInfo is obtained 
      from createPortInfo. Returns nonzero if addProvidesPort fails,
      for example, because that name is already provided.
       @see Services.java; UNADOPTED C++ definition of "an error occurs". */
    virtual int addProvidesPort(Port *inPort, PortInfo *name){};

      /** Notifies the framework that a previously exported Port is no longer 
	  available for use. */
    virtual void removeProvidesPort(CONST char *name){};
};

}}}
#endif

#endif




