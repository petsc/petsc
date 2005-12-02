#ifndef included_ALE_ALE_log_hh
#define included_ALE_ALE_log_hh

#include <petsc.h>

namespace ALE {
  typedef PetscCookie LogCookie;
  typedef PetscEvent  LogEvent;

  LogCookie LogCookieRegister(const char *name);

  int       LogStageRegister(const char *name);
  void      LogStagePush(int stage);
  void      LogStagePop(int stage);

  LogEvent  LogEventRegister(LogCookie cookie, const char* event_name);
  void      LogEventBegin(int e);
  void      LogEventEnd(int e);

} // namespace ALE

#endif
