
#include "petsc.h"


// --------------------------------------------------------------------------------------------------------

EXTERN_C_BEGIN
char **getESIFactoryList() {
  static char *list[] = {
    (char *) "create_esi_petsc_vectorfactory esi::petsc::Vector",
    (char *) "create_petra_esi_vectorfactory Petra_ESI_Vector",
    (char *) "create_esi_petsc_indexspacefactory esi::petsc::IndexSpace",
    (char *) "create_petra_esi_indexspacefactory Petra_ESI_IndexSpace",
    (char *) "create_esi_petsc_operatorfactory esi::petsc::Matrix",
    (char *) "create_petra_esi_operatorfactory Petra_ESI_CRS_Matrix",
    (char *) "create_esi_petsc_preconditionerfactory esi::petsc::Preconditioner",
    (char *) "create_esi_petsc_solveriterativefactory esi::petsc::SolverIterative",
    0};
  return list;
}
EXTERN_C_END


