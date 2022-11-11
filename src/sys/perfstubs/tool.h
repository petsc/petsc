// Copyright (c) 2019 University of Oregon
// Distributed under the BSD Software License
// (See accompanying file LICENSE.txt)

#pragma once
#include <stdint.h>

/****************************************************************************/
/* Declare the structures that a tool should use to return performance data. */
/****************************************************************************/

typedef struct ps_tool_timer_data
{
    unsigned int num_timers;
    unsigned int num_threads;
    unsigned int num_metrics;
    char **timer_names;
    char **metric_names;
    double *values;
} ps_tool_timer_data_t;

typedef struct ps_tool_counter_data
{
    unsigned int num_counters;
    unsigned int num_threads;
    char **counter_names;
    double *num_samples;
    double *value_total;
    double *value_min;
    double *value_max;
    double *value_sumsqr;
} ps_tool_counter_data_t;

typedef struct ps_tool_metadata
{
    unsigned int num_values;
    char **names;
    char **values;
} ps_tool_metadata_t;

/****************************************************************************/
/* Declare the typedefs of the functions that a tool should implement. */
/****************************************************************************/

/* Logistical functions */
typedef void  (*ps_initialize_t)(void);
typedef void  (*ps_finalize_t)(void);
typedef void  (*ps_register_thread_t)(void);
typedef void  (*ps_dump_data_t)(void);
/* Data entry functions */
typedef void* (*ps_timer_create_t)(const char *);
typedef void  (*ps_timer_start_t)(const void *);
typedef void  (*ps_timer_stop_t)(const void *);
typedef void  (*ps_set_parameter_t)(const char *, int64_t);
typedef void  (*ps_dynamic_phase_start_t)(const char *, int);
typedef void  (*ps_dynamic_phase_stop_t)(const char *, int);
typedef void* (*ps_create_counter_t)(const char *);
typedef void  (*ps_sample_counter_t)(const void *, double);
typedef void  (*ps_set_metadata_t)(const char *, const char *);
/* Data Query Functions */
typedef void  (*ps_get_timer_data_t)(ps_tool_timer_data_t *);
typedef void  (*ps_get_counter_data_t)(ps_tool_counter_data_t *);
typedef void  (*ps_get_metadata_t)(ps_tool_metadata_t *);
typedef void  (*ps_free_timer_data_t)(ps_tool_timer_data_t *);
typedef void  (*ps_free_counter_data_t)(ps_tool_counter_data_t *);
typedef void  (*ps_free_metadata_t)(ps_tool_metadata_t *);

/****************************************************************************/
/* Declare the structure used to register a tool */
/****************************************************************************/

typedef struct ps_plugin_data {
    char * tool_name;
    /* Logistical functions */
    ps_initialize_t initialize;
    ps_finalize_t finalize;
    ps_register_thread_t register_thread;
    ps_dump_data_t dump_data;
    /* Data entry functions */
    ps_timer_create_t timer_create;
    ps_timer_start_t timer_start;
    ps_timer_stop_t timer_stop;
    ps_set_parameter_t set_parameter;
    ps_dynamic_phase_start_t dynamic_phase_start;
    ps_dynamic_phase_stop_t dynamic_phase_stop;
    ps_create_counter_t create_counter;
    ps_sample_counter_t sample_counter;
    ps_set_metadata_t set_metadata;
    /* Data Query Functions */
    ps_get_timer_data_t get_timer_data;
    ps_get_counter_data_t get_counter_data;
    ps_get_metadata_t get_metadata;
    ps_free_timer_data_t free_timer_data;
    ps_free_counter_data_t free_counter_data;
    ps_free_metadata_t free_metadata;
} ps_plugin_data_t;

/****************************************************************************/
/* Declare the register/deregister weak symbols (implemented by the plugin API
 * for the registration process */
/****************************************************************************/

typedef int  (*ps_register_t)(ps_plugin_data_t *);
typedef void (*ps_deregister_t)(int);

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __linux__
extern __attribute__((weak)) int  ps_register_tool(ps_plugin_data_t * tool);
extern __attribute__((weak)) void ps_deregister_tool(int tool_id);
#else /* use _WIN32 or _WIN64 */
extern int  ps_register_tool(ps_plugin_data_t * tool);
extern void ps_deregister_tool(int tool_id);
#endif

#ifdef __cplusplus
}
#endif


