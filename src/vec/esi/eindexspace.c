
/*
      Makes a PETSc Map look like an esi::IndexSpace
*/

#include "esi/petsc/indexspace.h"


esi::petsc::IndexSpace<int>::IndexSpace(MPI_Comm comm, int n, int N)
{
  int ierr;
  ierr = PetscMapCreateMPI(comm,n,N,&this->map);
  this->pobject = (PetscObject)this->map;
  ierr = PetscObjectGetComm((PetscObject)this->map,&this->comm);
}

esi::petsc::IndexSpace<int>::IndexSpace(::esi::IndexSpace<int> &sourceIndexSpace)
{
  int      ierr,n,N;
  MPI_Comm *comm;

  ierr = sourceIndexSpace.getRunTimeModel("MPI",static_cast<void *>(comm));
  ierr = sourceIndexSpace.getGlobalSize(N);
  {
    ::esi::IndexSpace<int> *amap;

    ierr = sourceIndexSpace.getInterface("esi::IndexSpace",static_cast<void *>(amap));
    ierr = amap->getLocalSize(n);
  }
  ierr = PetscMapCreateMPI(*comm,n,N,&this->map);
  this->pobject = (PetscObject)this->map;
  ierr = PetscObjectGetComm((PetscObject)this->map,&this->comm);
}

esi::petsc::IndexSpace<int>::IndexSpace(PetscMap sourceIndexSpace)
{
  PetscObjectReference((PetscObject) sourceIndexSpace);
  this->map = sourceIndexSpace;
  this->pobject = (PetscObject)this->map;
  PetscObjectGetComm((PetscObject)sourceIndexSpace,&this->comm);
}

esi::petsc::IndexSpace<int>::~IndexSpace()
{
  int ierr;
  ierr = PetscMapDestroy(this->map); 
}

/* ---------------esi::Object methods ------------------------------------------------------------ */
::esi::ErrorCode esi::petsc::IndexSpace<int>::getInterface(const char* name, void *& iface)
{
  PetscTruth flg;
  if (PetscStrcmp(name,"esi::Object",&flg),flg){
    iface = (void *) (::esi::Object *) this;
  } else if (PetscStrcmp(name,"esi::IndexSpace",&flg),flg){
    iface = (void *) (::esi::IndexSpace<int> *) this;
  } else if (PetscStrcmp(name,"esi::petsc::IndexSpace",&flg),flg){
    iface = (void *) (::esi::petsc::IndexSpace<int> *) this;
  } else {
    iface = 0;
  }
  return 0;
}

::esi::ErrorCode esi::petsc::IndexSpace<int>::getInterfacesSupported(::esi::Argv * list)
{
  list->appendArg("esi::Object");
  list->appendArg("esi::IndexSpace");
  list->appendArg("esi::petsc::IndexSpace");
  return 0;
}


/* -------------- esi::IndexSpace methods --------------------------------------------*/
::esi::ErrorCode esi::petsc::IndexSpace<int>::getGlobalSize(int &globalSize)
{
  return PetscMapGetSize(this->map,&globalSize);
}

::esi::ErrorCode esi::petsc::IndexSpace<int>::getLocalSize(int &localSize)
{
  return PetscMapGetLocalSize(this->map,&localSize);
}

::esi::ErrorCode esi::petsc::IndexSpace<int>::getLocalPartitionOffset(int &localoffset)
{ 
  return PetscMapGetLocalRange(this->map,&localoffset,PETSC_IGNORE);
}

::esi::ErrorCode esi::petsc::IndexSpace<int>::getGlobalPartitionOffsets(int *globaloffsets)
{ 
  int ierr,*iglobaloffsets;
  int size;   

  ierr = PetscMapGetGlobalRange(this->map,&iglobaloffsets);
  ierr = MPI_Comm_size(this->comm,&size);
  ierr = PetscMemcpy(globaloffsets,iglobaloffsets,(size+1)*sizeof(int));
  return ierr;
}

::esi::ErrorCode esi::petsc::IndexSpace<int>::getGlobalPartitionSizes(int *globalsizes)
{ 
  int ierr,i,n,*globalranges;


  ierr = MPI_Comm_size(this->comm,&n);
  ierr = PetscMapGetGlobalRange(this->map,&globalranges);
  for (i=0; i<n; i++) {
    globalsizes[i] = globalranges[i+1] - globalranges[i];
  }
  return 0;
}

  /* -------------------------------------------------------------------------*/
namespace esi{namespace petsc{

template<class Ordinal> class IndexSpaceFactory : public virtual ::esi::IndexSpaceFactory<Ordinal>
{
  public:

    // Destructor.
  virtual ~IndexSpaceFactory(void){};

    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *svc)
    {
      svc->addProvidesPort(this,svc->createPortInfo("getIndexSpace", "esi::IndexSpaceFactory", 0));
    };
#endif

    // Construct a IndexSpace
    virtual ::esi::ErrorCode getIndexSpace(const char * name,void *comm,int m,::esi::IndexSpace<Ordinal>*&v)
    {
      PetscTruth ismpi;
      int ierr = PetscStrcmp(name,"MPI",&ismpi);CHKERRQ(ierr);
      if (!ismpi) SETERRQ1(1,"%s not supported, only MPI supported as RunTimeModel",name);
      v = new esi::petsc::IndexSpace<Ordinal>(*(MPI_Comm*)comm,m,PETSC_DETERMINE);
      return 0;
    };

};
}}
EXTERN_C_BEGIN
#if defined(PETSC_HAVE_CCA)
gov::cca::Component *create_esi_petsc_indexspacefactory(void)
{
  return dynamic_cast<gov::cca::Component *>(new esi::petsc::IndexSpaceFactory<int>);
}
#else
void *create_esi_petsc_indexspacefactory(void)
{
  return (void *)(new esi::petsc::IndexSpaceFactory<int>);
}
#endif
EXTERN_C_END

#if defined(PETSC_HAVE_TRILINOS)
#define PETRA_MPI /* used by Ptera to indicate MPI code */
#include "Petra_ESI_IndexSpace.h"

template<class Ordinal> class Petra_ESI_IndexSpaceFactory : public virtual ::esi::IndexSpaceFactory<Ordinal>
{
  public:

    // Destructor.
  virtual ~Petra_ESI_IndexSpaceFactory(void){};

    // Interface for gov::cca::Component
#if defined(PETSC_HAVE_CCA)
    virtual void setServices(gov::cca::Services *svc)
    {
      svc->addProvidesPort(this,svc->createPortInfo("getIndexSpace", "esi::IndexSpaceFactory", 0));
    };
#endif

    // Construct a IndexSpace
    virtual ::esi::ErrorCode getIndexSpace(const char * name,void *comm,int m,::esi::IndexSpace<Ordinal>*&v)
    {
      PetscTruth ismpi;
      int ierr = PetscStrcmp(name,"MPI",&ismpi);CHKERRQ(ierr);
      if (!ismpi) SETERRQ1(1,"%s not supported, only MPI supported as RunTimeModel",name);
      Petra_Comm *pcomm = new Petra_Comm(*(MPI_Comm*)comm);
      v = new Petra_ESI_IndexSpace<Ordinal>(-1,m,0,*pcomm); 
      if (!v) SETERRQ(1,"Unable to create Petra_ESI_IndexSpace");
      return 0;
    };

};
EXTERN_C_BEGIN
#if defined(PETSC_HAVE_CCA)
gov::cca::Component *create_petra_esi_indexspacefactory(void)
{
  return dynamic_cast<gov::cca::Component *>(new Petra_ESI_IndexSpaceFactory<int>);
}
#else
void *create_petra_esi_indexspacefactory(void)
{
  return (void *)(new Petra_ESI_IndexSpaceFactory<int>);
}
#endif
EXTERN_C_END
#endif
