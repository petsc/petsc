
/*
      Makes a PETSc Map look like an esi::IndexSpace
*/

#include "esi/petsc/indexspace.h"

esi::petsc::IndexSpace<int>::IndexSpace(MPI_Comm icomm, int n, int N)
{
  int ierr;
  ierr = PetscMapCreateMPI(icomm,n,N,&this->map);if (ierr) return;
  ierr = PetscObjectGetComm((PetscObject)this->map,&this->comm);if (ierr) return;
}

esi::petsc::IndexSpace<int>::IndexSpace(::esi::IndexSpace<int> &sourceIndexSpace)
{
  int      ierr,n,N;
  MPI_Comm *icomm;

  ierr = sourceIndexSpace.getRunTimeModel("MPI",reinterpret_cast<void *&>(icomm));if (ierr) return;
  ierr = sourceIndexSpace.getGlobalSize(N);if (ierr) return;
  {
    ::esi::IndexSpace<int> *amap;

    ierr = sourceIndexSpace.getInterface("esi::IndexSpace",reinterpret_cast<void *&>(amap));if (ierr) return;
    ierr = amap->getLocalSize(n);if (ierr) return;
  }
  ierr = PetscMapCreateMPI(*icomm,n,N,&this->map);if (ierr) return;
  ierr = PetscObjectGetComm((PetscObject)this->map,&this->comm);if (ierr) return;
}

esi::petsc::IndexSpace<int>::IndexSpace(PetscMap sourceIndexSpace)
{
  PetscObjectReference((PetscObject) sourceIndexSpace);
  this->map = sourceIndexSpace;
  PetscObjectGetComm((PetscObject)sourceIndexSpace,&this->comm);
}

esi::petsc::IndexSpace<int>::~IndexSpace()
{
  PetscMapDestroy(this->map);
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
  } else if (PetscStrcmp(name,"PetscMap",&flg),flg){
    iface = (void *) this->map;
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
  list->appendArg("PetscMap");
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

  ierr = PetscMapGetGlobalRange(this->map,&iglobaloffsets);CHKERRQ(ierr);
  ierr = MPI_Comm_size(this->comm,&size);CHKERRQ(ierr);
  ierr = PetscMemcpy(globaloffsets,iglobaloffsets,(size+1)*sizeof(int));CHKERRQ(ierr);
  return 0;
}

::esi::ErrorCode esi::petsc::IndexSpace<int>::getGlobalPartitionSizes(int *globalsizes)
{ 
  int ierr,i,n,*globalranges;


  ierr = MPI_Comm_size(this->comm,&n);CHKERRQ(ierr);
  ierr = PetscMapGetGlobalRange(this->map,&globalranges);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    globalsizes[i] = globalranges[i+1] - globalranges[i];
  }
  return 0;
}

  /* -------------------------------------------------------------------------*/

    // Construct a IndexSpace
::esi::ErrorCode esi::petsc::IndexSpace<int>::Factory::create(const char * name,void *comm,int m,int N,int base,::esi::IndexSpace<int>*&v)
{
  PetscTruth ismpi;
  int        ierr = PetscStrcmp(name,"MPI",&ismpi);CHKERRQ(ierr);
  if (!ismpi) SETERRQ1(1,"%s not supported, only MPI supported as RunTimeModel",name);
  v = new esi::petsc::IndexSpace<int>(*(MPI_Comm*)comm,m,N);
  return 0;
};

EXTERN_C_BEGIN
::esi::IndexSpace<int>::Factory *create_esi_petsc_indexspacefactory(void)
{
  return dynamic_cast< ::esi::IndexSpace<int>::Factory* >(new esi::petsc::IndexSpace<int>::Factory);
}
EXTERN_C_END

/* ::esi::petsc::IndexSpaceFactory<int> ISFInstForIntel64CompilerBug; */

#if defined(PETSC_HAVE_TRILINOS)
#define PETRA_MPI /* used by Ptera to indicate MPI code */
#include "Petra_ESI_IndexSpace.h"

template<class Ordinal> class Petra_ESI_IndexSpaceFactory : public virtual ::esi::IndexSpace<Ordinal>::Factory
{
  public:

    // Destructor.
  virtual ~Petra_ESI_IndexSpaceFactory(void){};

    // Construct a IndexSpace
    virtual ::esi::ErrorCode create(const char * name,void *comm,int m,int M,int base,::esi::IndexSpace<Ordinal>*&v)
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
::esi::IndexSpace<int>Factory *create_petra_esi_indexspacefactory(void)
{
  return dynamic_cast< ::esi::IndexSpace<int>Factory *>(new Petra_ESI_IndexSpaceFactory<int>);
}
EXTERN_C_END
#endif
