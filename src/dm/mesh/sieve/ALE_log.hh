#ifndef included_ALE_ALE_log_hh
#define included_ALE_ALE_log_hh

#include <petsc.h>

namespace ALE {

  void LogEventRegister(PetscEvent *event_ptr, const char* event_name, PetscCookie cookie);
  void LogEventBegin(int e);
  void LogEventEnd(int e);

} // namespace ALE

#endif
