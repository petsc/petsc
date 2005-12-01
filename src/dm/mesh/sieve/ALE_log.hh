#ifndef included_ALE_ALE_log_hh
#define included_ALE_ALE_log_hh

#include <petsc.h>

namespace ALE {
  typedef PetscCookie LogCookie;
  typedef PetscEvent  LogEvent;

  void LogCookieRegister(const char *name, LogCookie *cookie);

  void LogStageRegister(const char *name, int *stage);
  void LogStagePush(int stage);
  void LogStagePop(int stage);

  void LogEventRegister(LogCookie cookie, const char* event_name, LogEvent *event_ptr);
  void LogEventBegin(int e);
  void LogEventEnd(int e);

} // namespace ALE

#endif
