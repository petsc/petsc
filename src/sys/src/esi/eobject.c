

/*
    Provides several of the esi::Object methods used by all 
  of the esi::petsc classes
*/

#include "esi/petsc/object.h"


esi::ErrorCode esi::petsc::Object::getRunTimeModel(const char* name, void *& icomm)
{
  esi::ErrorCode ierr = 0;

  PetscTruth flg;
  if (PetscStrcmp(name,"MPI",&flg),flg){
    icomm = &(this->comm);
    return ierr;
  }
  icomm = 0;
  return 0;
}

esi::ErrorCode esi::petsc::Object::setRunTimeModel(const char* name, void * icomm)
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
  int ierr = 0;
  this->refcnt--;
  if (this->refcnt <= 0) delete this;
  return ierr;
}

#if defined(PETSC_HAVE_CCA)
void esi::petsc::Object::setServices(gov::cca::Services *)
{
  ;
}
#endif
