#define PETSC_DLL

#include "petscsys.h"
#include <sys/time.h>
#include <sys/systemcfg.h>

#undef __FUNCT__  
#define __FUNCT__ "rs6000_time"
PetscLogDouble PETSC_DLLEXPORT rs6000_time(void) 
{
   timebasestruct_t t;
   PetscLogDouble   time;

   PetscFunctionBegin;

   /* read in the register values */
   read_real_time(&t,TIMEBASE_SZ);
 
   /*
    * Call the conversion routines unconditionally, to ensure
    * that both values are in seconds and nanoseconds regardless
    * of the hardware platform. 
    */
   time_base_to_time(&t,TIMEBASE_SZ);
 
   time = t.tb_high + t.tb_low*1.0e-9;
   PetscFunctionReturn(time);
}
