
/* This file contains info for the use of PETSc Fortran interface stubs */

       int     PetscDoubleAddressToFortran(double*,double*);
       double *PetscDoubleAddressFromFortran(double*,int);
       int     PetscIntAddressToFortran(int*,int*);
       int    *PetscIntAddressFromFortran(int*,int); 
extern void   *PetscNull_Fortran;

#ifdef HAVE_64BITS
extern void *MPIR_ToPointer(int);
extern int MPIR_FromPointer(void*);
extern void MPIR_RmPointer(int);
#else
#define MPIR_ToPointer(a) (a)
#define MPIR_FromPointer(a) (int)(a)
#define MPIR_RmPointer(a)
#endif
