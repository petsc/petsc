
#include "petsc.h"


// --------------------------------------------------------------------------------------------------------

// CCAFFEINE expects each .so file to have a getComponentList function.
// See dccafe/cxx/dc/framework/ComponentFactory.h for details.
EXTERN_C_BEGIN
char **getComponentList() {
  static char *list[] = {
    "create_esi_petsc_vectorfactory esi::petsc::Vector",
    "create_petra_esi_vectorfactory Petra_ESI_Vector",
    "create_esi_petsc_indexspacefactory esi::petsc::IndexSpace",
    "create_petra_esi_indexspacefactory Petra_ESI_IndexSpace",
    "create_esi_petsc_operatorfactory esi::petsc::Matrix",
    "create_petra_esi_operatorfactory Petra_ESI_CRS_Matrix",
    "create_esi_petsc_preconditionerfactory esi::petsc::Preconditioner",
    "create_esi_petsc_solveriterativefactory esi::petsc::SolverIterative",
    0};
  return list;
}
EXTERN_C_END


