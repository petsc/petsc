
/*
      Interfaces the ESI_Map and ESI_MapPartition classes to the PETSc
    Map object class.
*/

#include "esi/petsc/map.h"


esi::petsc::Map<int>::Map(MPI_Comm comm, int n, int N)
{
  int ierr;
  ierr = PetscMapCreateMPI(comm,n,N,&this->map);
  this->pobject = (PetscObject)this->map;
  ierr = PetscObjectGetComm((PetscObject)this->map,&this->comm);
}

esi::petsc::Map<int>::Map(esi::Map<int> &sourceMap)
{
  int      ierr,n,N;
  MPI_Comm *comm;

  ierr = sourceMap.getRunTimeModel("MPI",static_cast<void *>(comm));
  ierr = sourceMap.getGlobalSize(N);
  {
    esi::MapPartition<int> *amap;

    ierr = sourceMap.getInterface("esi::MapPartition",static_cast<void *>(amap));
    ierr = amap->getLocalSize(n);
  }
  ierr = PetscMapCreateMPI(*comm,n,N,&this->map);
  this->pobject = (PetscObject)this->map;
  ierr = PetscObjectGetComm((PetscObject)this->map,&this->comm);
}

esi::petsc::Map<int>::Map(PetscMap sourceMap)
{
  PetscObjectReference((PetscObject) sourceMap);
  this->map = sourceMap;
  this->pobject = (PetscObject)this->map;
  PetscObjectGetComm((PetscObject)sourceMap,&this->comm);
}

esi::petsc::Map<int>::~Map()
{
  int ierr;
}

/* ---------------esi::Object methods ------------------------------------------------------------ */
esi::ErrorCode esi::petsc::Map<int>::getInterface(const char* name, void *& iface)
{
  PetscTruth flg;
  if (PetscStrcmp(name,"esi::Object",&flg),flg){
    iface = (void *) (esi::Object *) this;
  } else if (PetscStrcmp(name,"esi::Map",&flg),flg){
    iface = (void *) (esi::Map<int> *) this;
  } else if (PetscStrcmp(name,"esi::MapPartition",&flg),flg){
    iface = (void *) (esi::MapPartition<int> *) this;
  } else if (PetscStrcmp(name,"esi::petsc::Map",&flg),flg){
    iface = (void *) (esi::petsc::Map<int> *) this;
  } else {
    iface = 0;
  }
  return 0;
}

esi::ErrorCode esi::petsc::Map<int>::getInterfacesSupported(esi::Argv * list)
{
  list->appendArg("esi::Object");
  list->appendArg("esi::Map");
  list->appendArg("esi::MapPartition");
  list->appendArg("esi::petsc::Map");
  return 0;
}


/* -------------- esi::Map methods --------------------------------------------*/
esi::ErrorCode esi::petsc::Map<int>::getGlobalSize(int &globalSize)
{
  return PetscMapGetSize(this->map,&globalSize);
}

esi::ErrorCode esi::petsc::Map<int>::getLocalSize(int &localSize)
{
  return PetscMapGetLocalSize(this->map,&localSize);
}

/* -------------- esi::MapPartition methods --------------------------------------------*/
esi::ErrorCode esi::petsc::Map<int>::getLocalPartitionOffset(int &localoffset)
{ 
  return PetscMapGetLocalRange(this->map,&localoffset,PETSC_IGNORE);
}

esi::ErrorCode esi::petsc::Map<int>::getGlobalPartitionOffsets(int *globaloffsets)
{ 
  int ierr,*iglobaloffsets;
  ierr = PetscMapGetGlobalRange(this->map,&iglobaloffsets);
  int size;  ierr = PetscMemcpy(globaloffsets,iglobaloffsets,(size+1)*sizeof(int));
  return ierr;
}

esi::ErrorCode esi::petsc::Map<int>::getGlobalPartitionSizes(int *globalsizes)
{ 
  int ierr,i,n,*globalranges;


  ierr = MPI_Comm_size(this->comm,&n);
  ierr = PetscMapGetGlobalRange(this->map,&globalranges);
  for (i=0; i<n; i++) {
    globalsizes[i] = globalranges[i+1] - globalranges[i];
  }
  return 0;
}
