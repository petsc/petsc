#ifndef included_ALE_ALE_log_hh
#define included_ALE_ALE_log_hh

#include <petscsys.h>

namespace ALE {
  int  getVerbosity();
  void setVerbosity(const int& verbosity);

  typedef PetscClassId LogCookie;
  typedef int         LogStage;
  typedef PetscLogEvent  LogEvent;

  LogCookie LogCookieRegister(const char *name);

  LogStage  LogStageRegister(const char *name);
  void      LogStagePush(LogStage stage);
  void      LogStagePop(LogStage stage);

  LogEvent  LogEventRegister(LogCookie cookie, const char* event_name);
  LogEvent  LogEventRegister(const char* event_name);
  void      LogEventBegin(LogEvent e);
  void      LogEventEnd(LogEvent e);



} // namespace ALE

//    Helper macros that push and pop log stages bracketing method invocations.
// These depend on the __FUNCT__ macro being declared correctly -- as the qualified method name (e.g., PreSieve::cone).
//    Every ALE_LOG_STAGE_BEGIN must be matched by a corresponding ALE_LOG_STAGE_END.
// For proper logging, these macro calls must be placed outside of all code in a function, including variable declaration,
// except return value declaration and the actual return statement. This might require some code rearrangement.
// In particular, returns from inside the block bracketed by the macros will break the stage stack.
//    ALE_LOG_STAGE_START and ALE_LOG_STAGE_FINISH mirror the corresponding BEGIN and END macros, except that they do not contain
// opening and closing braces and can be used more freely throughout the code
//    ALE_LOG_EVENT_START and ALE_LOG_EVENT_FINISH can likewise be used throughout the code to start and stop logging of an event
// associate with the function __FUNCT__.  The difference between function stages and events is implementation-dependent
// (currently PETSc logging).

#if (defined ALE_USE_LOGGING) && (defined ALE_LOGGING_USE_STAGES)

#define ALE_LOG_STAGE_START                                 \
  {                                                         \
    ALE::LogStage stage = ALE::LogStageRegister(__FUNCT__); \
    ALE::LogStagePush(stage);                               \
  }

#define ALE_LOG_STAGE_FINISH                                \
  {                                                         \
    ALE::LogStage stage = ALE::LogStageRegister(__FUNCT__); \
    ALE::LogStagePop(stage);                                \
  }

#define ALE_LOG_STAGE_BEGIN    ALE_LOG_STAGE_START  {
#define ALE_LOG_STAGE_END      } ALE_LOG_STAGE_FINISH

#else

#define ALE_LOG_STAGE_START  {}
#define ALE_LOG_STAGE_FINISH {}
#define ALE_LOG_STAGE_BEGIN  {}
#define ALE_LOG_STAGE_END  {}

#endif

#if (defined ALE_USE_LOGGING) && (defined ALE_LOGGING_USE_EVENTS)

#define ALE_LOG_EVENT_BEGIN                                 \
  {                                                         \
    ALE::LogEvent event = ALE::LogEventRegister(__FUNCT__); \
    ALE::LogEventBegin(event);                              \
  }

#define ALE_LOG_EVENT_END                                   \
  {                                                         \
    ALE::LogEvent event = ALE::LogEventRegister(__FUNCT__); \
    ALE::LogEventEnd(event);                                \
  }

#else

#define ALE_LOG_EVENT_BEGIN  {}
#define ALE_LOG_EVENT_END    {}

#endif

#endif
