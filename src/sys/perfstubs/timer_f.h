! Copyright (c) 2019-2020 University of Oregon
! Distributed under the BSD Software License
! (See accompanying file LICENSE.txt)

#ifdef PERFSTUBS_USE_TIMERS

!
!    Macro API for option of entirely disabling at compile time
!    To use this API, set the Macro PERFSTUBS_USE_TIMERS on the command
!    line or in a config.h file, however your project does it
!

#define PERFSTUBS_INITIALIZE() call ps_initialize()
#define PERFSTUBS_FINALIZE() call ps_finalize()
#define PERFSTUBS_DUMP_DATA() call ps_dump_data()
#define PERFSTUBS_REGISTER_THREAD() call ps_register_thread()
#define PERFSTUBS_TIMER_CREATE(_timer_object, _timer_name) \
    call ps_timer_create_fortran(_timer_object, _timer_name//CHAR(0))
#define PERFSTUBS_TIMER_START(_timer_object) \
    call ps_timer_start_fortran(_timer_object)
#define PERFSTUBS_TIMER_STOP(_timer_object) \
    call ps_timer_stop_fortran(_timer_object)
#define PERFSTUBS_SET_PARAMETER(_parameter_name, parameter_value) \
    call ps_set_parameter(_parameter_name//CHAR(0), parameter_value)
#define PERFSTUBS_DYNAMIC_PHASE_START(_phase_prefix, _iteration_index) \
    call ps_dynamic_phase_start(_phase_prefix//CHAR(0), _iteration_index)
#define PERFSTUBS_DYNAMIC_PHASE_STOP(_phase_prefix, _iteration_index) \
    call ps_dynamic_phase_stop(_phase_prefix//CHAR(0), _iteration_index)
#define PERFSTUBS_CREATE_COUNTER(_counter_object, _name) \
    call ps_create_counter_fortran(_counter_object, _name//CHAR(0))
#define PERFSTUBS_SAMPLE_COUNTER(_counter_object, _value) \
    call ps_sample_counter_fortran(_counter_object, _value)
#define PERFSTUBS_METADATA(_name, _value) \
    call ps_set_metadata(_name//CHAR(0), _value//CHAR(0))

! // defined(PERFSTUBS_USE_TIMERS)
#else

#define PERFSTUBS_INIT()
#define PERFSTUBS_DUMP_DATA()
#define PERFSTUBS_REGISTER_THREAD()
#define PERFSTUBS_TIMER_CREATE(_timer_object, _timer_name)
#define PERFSTUBS_TIMER_START(_timer_object)
#define PERFSTUBS_TIMER_STOP(_timer_object)
#define PERFSTUBS_SET_PARAMETER(_parameter_name, _parameter_value)
#define PERFSTUBS_DYNAMIC_PHASE_START(_phase_prefix, _iteration_index)
#define PERFSTUBS_DYNAMIC_PHASE_STOP(_phase_prefix, _iteration_index)
#define PERFSTUBS_TIMER_START_FUNC()
#define PERFSTUBS_TIMER_STOP_FUNC()
#define PERFSTUBS_CREATE_COUNTER(_counter_object, _name)
#define PERFSTUBS_SAMPLE_COUNTER(_counter_object, _value)
#define PERFSTUBS_METADATA(_name, _value)

! // defined(PERFSTUBS_USE_TIMERS)
#endif

