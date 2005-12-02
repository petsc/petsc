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

// Helper macros that push and pop log stages bracketing method invocations.
// These depend on the __FUNCT__ macro being declared correctly -- as the qualified method name (e.g., PreSieve::cone).
// Every ALE_LOG_STAGE_BEGIN must be matched by a corresponding ALE_LOG_STAGE_END.
// For proper logging, these macro calls must be placed outside of all code in a function, including variable declaration,
// except return value declaration and the actual return statement. This might require some code rearrangement.
// In particular, returns from inside the block bracketed by the macros will break the stage stack.
#if (defined ALE_USE_LOGGING) && (defined ALE_LOGGING_USE_STAGES)
#define ALE_LOG_STAGE_BEGIN                                                             \
  {                                                                                     \
    int stage = LogStageRegister(__FUNCT__);                                            \
    LogStagePush(stage);                                                                \
  }                                                                                     \
  {                                                                               
#define ALE_LOG_STAGE_END                                                               \
  }                                                                                     \
  {                                                                                     \
    int stage = LogStageRegister(__FUNCT__);                                            \
    LogStagePop(stage);                                                                 \
  }                                                                                     
#else
#define ALE_LOG_STAGE_BEGIN {}
#define ALE_LOG_STAGE_BEGIN {}
#endif


#endif
