
#include "petsc.h"


// --------------------------------------------------------------------------------------------------------

// CCAFFEINE expects each .so file to have a getComponentList function.
// See dccafe/cxx/dc/framework/ComponentFactory.h for details.
EXTERN_C_BEGIN
char **getComponentList() {
  static char *list[8];
  list[0] = "create_esi_petsc_vectorfactory esi::petsc::Vector";
  list[1] = "create_petra_esi_vectorfactory Petra_ESI_Vector";
  list[2] = "create_esi_petsc_indexspacefactory esi::petsc::IndexSpace";
  list[3] = "create_petra_esi_indexspacefactory Petra_ESI_IndexSpace";
  list[4] = "create_esi_petsc_operatorfactory esi::petsc::Matrix";
  list[5] = "create_petra_esi_operatorfactory Petra_ESI_CRS_Matrix";
  list[6] = "create_esi_petsc_preconditionerfactory esi::petsc::Preconditioner";
  list[7] = 0;
  return list;
}
EXTERN_C_END


