// Copyright (c) 2019 University of Oregon
// Distributed under the BSD Software License
// (See accompanying file LICENSE.txt)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // needed to define RTLD_DEFAULT
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifndef PERFSTUBS_STANDALONE
#include "petscconf.h"
#ifdef PETSC_HAVE_DLFCN_H
#define PERFSTUBS_HAVE_DLFCN_H
#endif
#endif
#if defined(__linux__) && defined(PERFSTUBS_HAVE_DLFCN_H)
#include <dlfcn.h>
#else
#define PERFSTUBS_USE_STATIC 1
#endif
#define PERFSTUBS_USE_TIMERS
#include "timer.h"

#define MAX_TOOLS 1

#if defined(_WIN32)||defined(WIN32)||defined(_WIN64)||defined(WIN64)||defined(__CYGWIN__)||defined(__APPLE__)
#define PERFSTUBS_OFF
#endif

/* Make sure that the Timer singleton is constructed when the
 * library is loaded.  This will ensure (on linux, anyway) that
 * we can assert that we have m_Initialized on the main thread. */
//static void __attribute__((constructor)) initialize_library(void);

/* Globals for the plugin API */

int perfstubs_initialized = PERFSTUBS_UNKNOWN;
int num_tools_registered = 0;
/* Keep track of whether the thread has been registered */
#ifndef PERFSTUBS_OFF
__thread int thread_seen = 0;
#endif
/* Function pointers */

ps_initialize_t initialize_functions[MAX_TOOLS];
ps_register_thread_t register_thread_functions[MAX_TOOLS];
ps_finalize_t finalize_functions[MAX_TOOLS];
ps_dump_data_t dump_data_functions[MAX_TOOLS];
ps_timer_create_t timer_create_functions[MAX_TOOLS];
ps_timer_start_t timer_start_functions[MAX_TOOLS];
ps_timer_stop_t timer_stop_functions[MAX_TOOLS];
ps_set_parameter_t set_parameter_functions[MAX_TOOLS];
ps_dynamic_phase_start_t dynamic_phase_start_functions[MAX_TOOLS];
ps_dynamic_phase_stop_t dynamic_phase_stop_functions[MAX_TOOLS];
ps_create_counter_t create_counter_functions[MAX_TOOLS];
ps_sample_counter_t sample_counter_functions[MAX_TOOLS];
ps_set_metadata_t set_metadata_functions[MAX_TOOLS];
ps_get_timer_data_t get_timer_data_functions[MAX_TOOLS];
ps_get_counter_data_t get_counter_data_functions[MAX_TOOLS];
ps_get_metadata_t get_metadata_functions[MAX_TOOLS];
ps_free_timer_data_t free_timer_data_functions[MAX_TOOLS];
ps_free_counter_data_t free_counter_data_functions[MAX_TOOLS];
ps_free_metadata_t free_metadata_functions[MAX_TOOLS];

#ifndef PERFSTUBS_OFF

#ifdef PERFSTUBS_USE_STATIC

#if defined(__clang__) && defined(__APPLE__)
#define PS_WEAK_PRE
#define PS_WEAK_POST __attribute__((weak_import))
#define PS_WEAK_POST_NULL __attribute__((weak_import))
#else
#define PS_WEAK_PRE __attribute__((weak))
#define PS_WEAK_POST
#define PS_WEAK_POST_NULL
#endif

PS_WEAK_PRE void ps_tool_initialize(void) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_register_thread(void) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_finalize(void) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_dump_data(void) PS_WEAK_POST;
PS_WEAK_PRE void* ps_tool_timer_create(const char *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_timer_start(const void *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_timer_stop(const void *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_set_parameter(const char *, int64_t) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_dynamic_phase_start(const char *, int) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_dynamic_phase_stop(const char *, int) PS_WEAK_POST;
PS_WEAK_PRE void* ps_tool_create_counter(const char *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_sample_counter(const void *, double) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_set_metadata(const char *, const char *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_get_timer_data(ps_tool_timer_data_t *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_get_counter_data(ps_tool_counter_data_t *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_get_metadata(ps_tool_metadata_t *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_free_timer_data(ps_tool_timer_data_t *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_free_counter_data(ps_tool_counter_data_t *) PS_WEAK_POST;
PS_WEAK_PRE void ps_tool_free_metadata(ps_tool_metadata_t *) PS_WEAK_POST;
#endif


// Disable pedantic, see https://stackoverflow.com/a/36385690
#pragma GCC diagnostic push  // Save actual diagnostics state
#pragma GCC diagnostic ignored "-Wpedantic" // Disable pedantic

#endif //PERFSTUBS_OFF

void initialize_library(void) {
#ifndef PERFSTUBS_OFF
#ifdef PERFSTUBS_USE_STATIC
    /* The initialization function is the only required one */
    initialize_functions[0] = &ps_tool_initialize;
    if (initialize_functions[0] == NULL) {
        return;
    }
    printf("Found ps_tool_initialize(), registering tool\n");
    register_thread_functions[0] = &ps_tool_register_thread;
    finalize_functions[0] = &ps_tool_finalize;
    dump_data_functions[0] = &ps_tool_dump_data;
    timer_create_functions[0] = &ps_tool_timer_create;
    timer_start_functions[0] = &ps_tool_timer_start;
    timer_stop_functions[0] = &ps_tool_timer_stop;
    set_parameter_functions[0] = &ps_tool_set_parameter;
    dynamic_phase_start_functions[0] = &ps_tool_dynamic_phase_start;
    dynamic_phase_stop_functions[0] = &ps_tool_dynamic_phase_stop;
    create_counter_functions[0] = &ps_tool_create_counter;
    sample_counter_functions[0] = &ps_tool_sample_counter;
    set_metadata_functions[0] = &ps_tool_set_metadata;
    get_timer_data_functions[0] = &ps_tool_get_timer_data;
    get_counter_data_functions[0] = &ps_tool_get_counter_data;
    get_metadata_functions[0] = &ps_tool_get_metadata;
    free_timer_data_functions[0] = &ps_tool_free_timer_data;
    free_counter_data_functions[0] = &ps_tool_free_counter_data;
    free_metadata_functions[0] = &ps_tool_free_metadata;
#else
    initialize_functions[0] =
        (ps_initialize_t)dlsym(RTLD_DEFAULT, "ps_tool_initialize");
    if (initialize_functions[0] == NULL) {
        perfstubs_initialized = PERFSTUBS_FAILURE;
        return;
    }
    printf("Found ps_tool_initialize(), registering tool\n");
    finalize_functions[0] =
        (ps_finalize_t)dlsym(RTLD_DEFAULT, "ps_tool_finalize");
    register_thread_functions[0] =
        (ps_register_thread_t)dlsym(RTLD_DEFAULT, "ps_tool_register_thread");
    dump_data_functions[0] =
        (ps_dump_data_t)dlsym(RTLD_DEFAULT, "ps_tool_dump_data");
    timer_create_functions[0] =
        (ps_timer_create_t)dlsym(RTLD_DEFAULT,
        "ps_tool_timer_create");
    timer_start_functions[0] =
        (ps_timer_start_t)dlsym(RTLD_DEFAULT, "ps_tool_timer_start");
    timer_stop_functions[0] =
        (ps_timer_stop_t)dlsym(RTLD_DEFAULT, "ps_tool_timer_stop");
    set_parameter_functions[0] =
        (ps_set_parameter_t)dlsym(RTLD_DEFAULT, "ps_tool_set_parameter");
    dynamic_phase_start_functions[0] = (ps_dynamic_phase_start_t)dlsym(
        RTLD_DEFAULT, "ps_tool_dynamic_phase_start");
    dynamic_phase_stop_functions[0] = (ps_dynamic_phase_stop_t)dlsym(
        RTLD_DEFAULT, "ps_tool_dynamic_phase_stop");
    create_counter_functions[0] = (ps_create_counter_t)dlsym(
        RTLD_DEFAULT, "ps_tool_create_counter");
    sample_counter_functions[0] = (ps_sample_counter_t)dlsym(
        RTLD_DEFAULT, "ps_tool_sample_counter");
    set_metadata_functions[0] =
        (ps_set_metadata_t)dlsym(RTLD_DEFAULT, "ps_tool_set_metadata");
    get_timer_data_functions[0] = (ps_get_timer_data_t)dlsym(
        RTLD_DEFAULT, "ps_tool_get_timer_data");
    get_counter_data_functions[0] = (ps_get_counter_data_t)dlsym(
        RTLD_DEFAULT, "ps_tool_get_counter_data");
    get_metadata_functions[0] = (ps_get_metadata_t)dlsym(
        RTLD_DEFAULT, "ps_tool_get_metadata");
    free_timer_data_functions[0] = (ps_free_timer_data_t)dlsym(
        RTLD_DEFAULT, "ps_tool_free_timer_data");
    free_counter_data_functions[0] = (ps_free_counter_data_t)dlsym(
        RTLD_DEFAULT, "ps_tool_free_counter_data");
    free_metadata_functions[0] = (ps_free_metadata_t)dlsym(
        RTLD_DEFAULT, "ps_tool_free_metadata");
#endif
    perfstubs_initialized = PERFSTUBS_SUCCESS;
    /* Increment the number of tools */
    num_tools_registered = 1;
#endif //PERFSTUBS_OFF
}
#ifndef PERFSTUBS_OFF
#pragma GCC diagnostic pop  // Restore diagnostics state
#endif

char * ps_make_timer_name_(const char * file,
    const char * func, int line) {
    #ifndef PERFSTUBS_OFF
    /* The length of the line number as a string is floor(log10(abs(num))) */
    int string_length = (strlen(file) + strlen(func) + floor(log10(abs(line))) + 11);
    char * name = (char*)calloc(string_length, sizeof(char));
    sprintf(name, "%s [{%s} {%d,0}]", func, file, line);
    return (name);
    #else
    return NULL;
    #endif
}

// used internally to the class
void ps_register_thread_internal(void) {
#ifndef PERFSTUBS_OFF
    	int i;
    for (i = 0 ; i < num_tools_registered ; i++) {
        register_thread_functions[i]();
    }
    thread_seen = 1;
#endif
}

/* Initialization */
void ps_initialize_(void) {
#ifndef PERFSTUBS_OFF
    int i;
    initialize_library();
    for (i = 0 ; i < num_tools_registered ; i++) {
        initialize_functions[i]();
    }
    /* No need to register the main thread */
    thread_seen = 1;
#endif
}

void ps_finalize_(void) {
    #ifndef PERFSTUBS_OFF
    int i;
    for (i = 0 ; i < num_tools_registered ; i++) {
        finalize_functions[i]();
    }
    #endif
}

void ps_register_thread_(void) {
#ifndef PERFSTUBS_OFF	
    if (thread_seen == 0) {
        ps_register_thread_internal();
    }
#endif
}

void* ps_timer_create_(const char *timer_name) {
    #ifndef PERFSTUBS_OFF
    void ** objects = (void**)calloc(num_tools_registered, sizeof(void*));
    int i;
    for (i = 0 ; i < num_tools_registered ; i++) {
        objects[i] = (void*)timer_create_functions[i](timer_name);
    }
    return (void*)(objects);
    #else
    return NULL;
    #endif
}

void ps_timer_create_fortran_(void ** object, const char *timer_name) {
    #ifndef PERFSTUBS_OFF
    *object = ps_timer_create_(timer_name);
    #endif
}

void ps_timer_start_(const void *timer) {
    #ifndef PERFSTUBS_OFF
    void ** objects = (void**)(timer);
    int i;
    for (i = 0; i < num_tools_registered ; i++) {
        timer_start_functions[i](objects[i]);
    }
    #endif
}

void ps_timer_start_fortran_(const void **timer) {
    #ifndef PERFSTUBS_OFF
    ps_timer_start_(*timer);
    #endif
}

void ps_timer_stop_(const void *timer) {
    #ifndef PERFSTUBS_OFF
    void ** objects = (void**)(timer);
    int i;
    for (i = 0; i < num_tools_registered ; i++) {
        timer_stop_functions[i](objects[i]);
    }
    #endif
}

void ps_timer_stop_fortran_(const void **timer) {
    #ifndef PERFSTUBS_OFF
    ps_timer_stop_(*timer);
    #endif
}

void ps_set_parameter_(const char * parameter_name, int64_t parameter_value) {
    #ifndef PERFSTUBS_OFF
    int i;
    for (i = 0; i < num_tools_registered ; i++) {
        set_parameter_functions[i](parameter_name, parameter_value);
    }
    #endif
}

void ps_dynamic_phase_start_(const char *phase_prefix, int iteration_index) {
    #ifndef PERFSTUBS_OFF
    int i;
    for (i = 0; i < num_tools_registered ; i++) {
        dynamic_phase_start_functions[i](phase_prefix, iteration_index);
    }
    #endif
}

void ps_dynamic_phase_stop_(const char *phase_prefix, int iteration_index) {
    #ifndef PERFSTUBS_OFF
    int i;
    for (i = 0; i < num_tools_registered ; i++) {
        dynamic_phase_stop_functions[i](phase_prefix, iteration_index);
    }
    #endif
}

void* ps_create_counter_(const char *name) {
    #ifndef PERFSTUBS_OFF
    void ** objects = (void**)calloc(num_tools_registered, sizeof(void*));
    int i;
    for (i = 0 ; i < num_tools_registered ; i++) {
        objects[i] = (void*)create_counter_functions[i](name);
    }
    return (void*)(objects);
    #else
    return NULL;
    #endif
}

void ps_create_counter_fortran_(void ** object, const char *name) {
    #ifndef PERFSTUBS_OFF
    *object = ps_create_counter_(name);
    #endif
}

void ps_sample_counter_(const void *counter, const double value) {
    #ifndef PERFSTUBS_OFF
    void ** objects = (void**)(counter);
    int i;
    for (i = 0; i < num_tools_registered ; i++) {
        sample_counter_functions[i](objects[i], value);
    }
    #endif
}

void ps_sample_counter_fortran_(const void **counter, const double value) {
    #ifndef PERFSTUBS_OFF
    ps_sample_counter_(*counter, value);
    #endif
}

void ps_set_metadata_(const char *name, const char *value) {
    #ifndef PERFSTUBS_OFF
    int i;
    for (i = 0; i < num_tools_registered ; i++) {
        set_metadata_functions[i](name, value);
    }
    #endif
}

void ps_dump_data_(void) {
    #ifndef PERFSTUBS_OFF
    int i;
    for (i = 0; i < num_tools_registered ; i++) {
        dump_data_functions[i]();
    }
    #endif
}

void ps_get_timer_data_(ps_tool_timer_data_t *timer_data, int tool_id) {
    #ifndef PERFSTUBS_OFF
    if (tool_id < num_tools_registered) {
        get_timer_data_functions[tool_id](timer_data);
    }
    #endif
}

void ps_get_counter_data_(ps_tool_counter_data_t *counter_data, int tool_id) {
    #ifndef PERFSTUBS_OFF
    if (tool_id < num_tools_registered) {
        get_counter_data_functions[tool_id](counter_data);
    }
    #endif
}

void ps_get_metadata_(ps_tool_metadata_t *metadata, int tool_id) {
    #ifndef PERFSTUBS_OFF
    if (tool_id < num_tools_registered) {
        get_metadata_functions[tool_id](metadata);
    }
    #endif
}

void ps_free_timer_data_(ps_tool_timer_data_t *timer_data, int tool_id) {
    #ifndef PERFSTUBS_OFF
    if (tool_id < num_tools_registered) {
        free_timer_data_functions[tool_id](timer_data);
    }
    #endif
}

void ps_free_counter_data_(ps_tool_counter_data_t *counter_data, int tool_id) {
    #ifndef PERFSTUBS_OFF
    if (tool_id < num_tools_registered) {
        free_counter_data_functions[tool_id](counter_data);
    }
    #endif
}

void ps_free_metadata_(ps_tool_metadata_t *metadata, int tool_id) {
    #ifndef PERFSTUBS_OFF
    if (tool_id < num_tools_registered) {
        free_metadata_functions[tool_id](metadata);
    }
    #endif
}
