/* $Id: sseenabled.c,v 1.4 2001/04/13 19:10:40 buschelm Exp bsmith $ */
#define "petsc.h"

#ifdef PETSC_HAVE_ICL

/* Processor specific version for PentiumIII and Pentium4 */
__declspec(cpu_specific(pentium_iii))
int PetscSSEIsEnabled(PetscTruth *flag) {
  flag = PETSC_TRUE;
  return(0);
}

__declspec(cpu_specific(pentium_iii_no_xmm_regs))
int PetscSSEIsEnabled(PetscTruth *flag) {
  flag = PETSC_FALSE;
  return(0);
}
/* Generic Intel processor version (i.e., not PIII,P4) */
__declspec(cpu_specific(generic))
int PetscSSEIsEnabled(PetscTruth *flag) {
  flag = PETSC_FALSE;
  return(0);
}

/* Dummy stub performs the dispatch of appropriate version */ 
__declspec(cpu_dispatch(generic,pentium_iii_no_xmm_regs,pentium_iii))
int PetscSSEIsEnabled(void) {}

#else

/* Version to use if not compiling with ICL */
int PetscSSEIsEnabled(PetscTruth *flag) {
  flag = PETSC_FALSE;
  return(0);
}

#endif
