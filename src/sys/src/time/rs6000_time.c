#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: nt_time.c,v 1.12 1998/12/17 21:56:35 balay Exp $";
#endif

#include <petsc.h>
#if defined (USE_READ_REAL_TIME)
#include <sys/time.h>
#include <sys/systemcfg.h>

#undef __FUNC__  
#define __FUNC__ "rs6000_time"
PlogDouble rs6000_time(void) 
{
   timebasestruct_t t;
   PLogDouble time;

   PetscFunctionBegin();

   /* read in the register values */
   read_real_time(&t, TIMEBASE_SZ);
 
   /*
    * Call the conversion routines unconditionally, to ensure
    * that both values are in seconds and nanoseconds regardless
    * of the hardware platform. 
    */
   time_base_to_time(&t, TIMEBASE_SZ);
 
   time = t.tb_high + t.tb_low*1.0e-9;
   PetscFunctionReturn(time);
}
#endif
