// Copyright (c) 2019 University of Oregon
// Distributed under the BSD Software License
// (See accompanying file LICENSE.txt)

#pragma once
#define PERFSTUBS_USE_TIMERS

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "config.h"
#include "tool.h"

/* These macros help generate unique variable names within a function
 * based on the source code line number */
#define CONCAT_(x,y) x##y
#define CONCAT(x,y) CONCAT_(x,y)

/* ------------------------------------------------------------------ */
/* Define the C API and PerfStubs glue class first */
/* ------------------------------------------------------------------ */

/* Pretty functions will include the return type and argument types,
 * not just the function name.  If the compiler doesn't support it,
 * just use the function name. */

#if defined(__GNUC__)
#define __PERFSTUBS_FUNCTION__ __PRETTY_FUNCTION__
#else
#define __PERFSTUBS_FUNCTION__ __func__
#endif

#define PERFSTUBS_UNKNOWN 0
#define PERFSTUBS_SUCCESS 1
#define PERFSTUBS_FAILURE 2
#define PERFSTUBS_FINALIZED 3

extern int perfstubs_initialized;

/* ------------------------------------------------------------------ */
/* Now define the C API */
/* ------------------------------------------------------------------ */

#if defined(PERFSTUBS_USE_TIMERS)

/* regular C API */

#ifdef __cplusplus
extern "C" {
#endif

void  ps_initialize_(void);
void  ps_finalize_(void);
void  ps_register_thread_(void);
void  ps_dump_data_(void);
void* ps_timer_create_(const char *timer_name);
void  ps_timer_create_fortran_(void ** object, const char *timer_name);
void  ps_timer_start_(const void *timer);
void  ps_timer_start_fortran_(const void **timer);
void  ps_timer_stop_(const void *timer);
void  ps_timer_stop_fortran_(const void **timer);
void  ps_set_parameter_(const char *parameter_name, int64_t parameter_value);
void  ps_dynamic_phase_start_(const char *phasePrefix, int iterationIndex);
void  ps_dynamic_phase_stop_(const char *phasePrefix, int iterationIndex);
void* ps_create_counter_(const char *name);
void  ps_create_counter_fortran_(void ** object, const char *name);
void  ps_sample_counter_(const void *counter, const double value);
void  ps_sample_counter_fortran_(const void **counter, const double value);
void  ps_set_metadata_(const char *name, const char *value);

/* data query API */

void  ps_get_timer_data_(ps_tool_timer_data_t *timer_data, int tool_id);
void  ps_get_counter_data_(ps_tool_counter_data_t *counter_data, int tool_id);
void  ps_get_metadata_(ps_tool_metadata_t *metadata, int tool_id);
void  ps_free_timer_data_(ps_tool_timer_data_t *timer_data, int tool_id);
void  ps_free_counter_data_(ps_tool_counter_data_t *counter_data, int tool_id);
void  ps_free_metadata_(ps_tool_metadata_t *metadata, int tool_id);

char* ps_make_timer_name_(const char * file, const char * func, int line);

#ifdef __cplusplus
}
#endif

/* Macro API for option of entirely disabling at compile time
 * To use this API, set the Macro PERFSTUBS_USE_TIMERS on the command
 * line or in a config.h file, however your project does it
 */

#define PERFSTUBS_INITIALIZE() ps_initialize_();

#define PERFSTUBS_FINALIZE() ps_finalize_();

#define PERFSTUBS_REGISTER_THREAD() ps_register_thread_();

#define PERFSTUBS_DUMP_DATA() ps_dump_data_();

#define PERFSTUBS_TIMER_START(_timer, _timer_name) \
    static void * _timer = NULL; \
    if (perfstubs_initialized == PERFSTUBS_SUCCESS) { \
        if (_timer == NULL) { \
            _timer = ps_timer_create_(_timer_name); \
        } \
        ps_timer_start_(_timer); \
    };

#define PERFSTUBS_TIMER_STOP(_timer) \
    if (perfstubs_initialized == PERFSTUBS_SUCCESS) ps_timer_stop_(_timer); \

#define PERFSTUBS_SET_PARAMETER(_parameter, _value) \
    if (perfstubs_initialized == PERFSTUBS_SUCCESS) ps_set_parameter_(_parameter, _value);

#define PERFSTUBS_DYNAMIC_PHASE_START(_phase_prefix, _iteration_index) \
    if (perfstubs_initialized == PERFSTUBS_SUCCESS) \
    ps_dynamic_phase_start_(_phase_prefix, _iteration_index);

#define PERFSTUBS_DYNAMIC_PHASE_STOP(_phase_prefix, _iteration_index) \
    if (perfstubs_initialized == PERFSTUBS_SUCCESS) \
    ps_dynamic_phase_stop_(_phase_prefix, _iteration_index);

#define PERFSTUBS_TIMER_START_FUNC(_timer) \
    static void * _timer = NULL; \
    if (perfstubs_initialized == PERFSTUBS_SUCCESS) { \
        if (_timer == NULL) { \
            char * tmpstr = ps_make_timer_name_(__FILE__, \
            __PERFSTUBS_FUNCTION__, __LINE__); \
            _timer = ps_timer_create_(tmpstr); \
            free(tmpstr); \
        } \
        ps_timer_start_(_timer); \
    };

#define PERFSTUBS_TIMER_STOP_FUNC(_timer) \
    if (perfstubs_initialized == PERFSTUBS_SUCCESS) ps_timer_stop_(_timer);

#define PERFSTUBS_SAMPLE_COUNTER(_name, _value) \
    static void * CONCAT(__var,__LINE__) =  NULL; \
    if (perfstubs_initialized == PERFSTUBS_SUCCESS) { \
        if (CONCAT(__var,__LINE__) == NULL) { \
            CONCAT(__var,__LINE__) = ps_create_counter_(_name); \
        } \
        ps_sample_counter_(CONCAT(__var,__LINE__), _value); \
    };

#define PERFSTUBS_METADATA(_name, _value) \
    if (perfstubs_initialized == PERFSTUBS_SUCCESS) ps_set_metadata_(_name, _value);

#else // defined(PERFSTUBS_USE_TIMERS)

#define PERFSTUBS_INITIALIZE()
#define PERFSTUBS_FINALIZE()
#define PERFSTUBS_REGISTER_THREAD()
#define PERFSTUBS_DUMP_DATA()
#define PERFSTUBS_TIMER_START(_timer, _timer_name)
#define PERFSTUBS_TIMER_STOP(_timer_name)
#define PERFSTUBS_SET_PARAMETER(_parameter, _value)
#define PERFSTUBS_DYNAMIC_PHASE_START(_phase_prefix, _iteration_index)
#define PERFSTUBS_DYNAMIC_PHASE_STOP(_phase_prefix, _iteration_index)
#define PERFSTUBS_TIMER_START_FUNC(_timer)
#define PERFSTUBS_TIMER_STOP_FUNC(_timer)
#define PERFSTUBS_SAMPLE_COUNTER(_name, _value)
#define PERFSTUBS_METADATA(_name, _value)

#endif // defined(PERFSTUBS_USE_TIMERS)

#ifdef __cplusplus

#if defined(PERFSTUBS_USE_TIMERS)

/*
 * We allow the namespace to be changed, so that different libraries
 * can include their own implementation and not have a namespace collision.
 * For example, library A and executable B could both include the 
 * perfstubs_api code in their source tree, and change the namespace
 * respectively, instead of linking in the perfstubs library.
 */

#if defined(PERFSTUBS_NAMESPACE)
#define PERFSTUBS_INTERNAL_NAMESPACE PERFSTUBS_NAMESPACE
#else
#define PERFSTUBS_INTERNAL_NAMESPACE perfstubs_profiler
#endif

#include <memory>
#include <sstream>
#include <string>

namespace external
{

namespace PERFSTUBS_INTERNAL_NAMESPACE
{

class ScopedTimer
{
private:
    const void * m_timer;

public:
    ScopedTimer(const void * timer) : m_timer(timer)
    {
        if (perfstubs_initialized == PERFSTUBS_SUCCESS) ps_timer_start_(m_timer);
    }
    ~ScopedTimer()
    {
        if (perfstubs_initialized == PERFSTUBS_SUCCESS) ps_timer_stop_(m_timer);
    }
};

} // namespace PERFSTUBS_INTERNAL_NAMESPACE

} // namespace external

namespace PSNS = external::PERFSTUBS_INTERNAL_NAMESPACE;

#define PERFSTUBS_SCOPED_TIMER(__name) \
    static void * CONCAT(__var,__LINE__) = ps_timer_create_(__name); \
    PSNS::ScopedTimer CONCAT(__var2,__LINE__)(CONCAT(__var,__LINE__));

/* The string created by ps_make_timer_name is a memory leak, but 
 * it is only created once per function, since it is called when the 
 * static variable is first initialized. */
#define PERFSTUBS_SCOPED_TIMER_FUNC() \
    static void * CONCAT(__var,__LINE__) = \
        ps_timer_create_(ps_make_timer_name_(__FILE__, \
        __PERFSTUBS_FUNCTION__, __LINE__)); \
    PSNS::ScopedTimer CONCAT(__var2,__LINE__)(CONCAT(__var,__LINE__));

#else // defined(PERFSTUBS_USE_TIMERS)

#define PERFSTUBS_SCOPED_TIMER(__name)
#define PERFSTUBS_SCOPED_TIMER_FUNC()

#endif // defined(PERFSTUBS_USE_TIMERS)

#endif // ifdef __cplusplus
