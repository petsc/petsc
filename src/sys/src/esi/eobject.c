
/*
    Provides several of the esi::Object methods used by all 
  of the esi::petsc classes
*/

#include "esi/petsc/object.h"


esi::ErrorCode esi::petsc::Object::getRunTimeModel(const char* name, void *& comm)
{
  esi::ErrorCode ierr = 0;

  PetscTruth flg;
  if (PetscStrcmp(name,"MPI",&flg),flg){
    comm = &this->comm;
    return ierr;
  }
  comm = 0;
  return 0;
}

esi::ErrorCode esi::petsc::Object::setRunTimeModel(const char* name, void * comm)
{
  return 1;
}

esi::ErrorCode esi::petsc::Object::getRunTimeModelsSupported(esi::Argv * list)
{
  list->appendArg("MPI");
  return 0;
}

esi::ErrorCode esi::petsc::Object::getInterfacesSupported(esi::Argv * list)
{
  list->appendArg("esi:Object");
  return 0;
}

esi::ErrorCode esi::petsc::Object::getInterface(const char* name, void *& iface)
{
  PetscTruth flg;
  if (PetscStrcmp(name,"esi::Object",&flg),flg){
    iface = (void *) (esi::Object *) this;
  } else {
    iface = 0;
  }
  return 0;
}


esi::ErrorCode esi::petsc::Object::addReference()
{
  int ierr = 0;
  this->refcnt++;
  if (this->pobject) ierr = PetscObjectReference(this->pobject);
  return ierr;
}

esi::ErrorCode esi::petsc::Object::deleteReference()
{
  int ierr;
  this->refcnt--;
  if (this->pobject) ierr = PetscObjectDereference(this->pobject);
  if (this->refcnt <= 0) delete this;
  return ierr;
}
