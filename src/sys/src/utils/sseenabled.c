#ifdef PETSC_HAVE_ICL

/* Processor specific version for PentiumIII and Pentium4 */
__declspec(cpu_specific(pentium_iii))
PetscTruth PetscSSEIsEnabled(void) {
  return(PETSC_TRUE);
}

/* Generic Intel processor version (i.e., not PIII,P4) */
__declspec(cpu_specific(generic))
PetscTruth PetscSSEIsEnabled(void) {
  return(PETSC_FALSE);
}

/* Dummy stub performs the dispatch of appropriate version */ 
__declspec(cpu_dispatch(generic,pentium_iii))
PetscTruth PetscSSEIsEnabled(void) {}

#else

/* Version to use if not compiling with ICL */
PetscTruth PetscSSEIsEnabled(void) {
  return(PETSC_FALSE);
}

#endif
