/* itcreate.c */
/* Fortran interface file */

#ifdef POINTER_64_BITS
extern void *__ToPointer();
extern int __FromPointer();
extern void __RmPointer();
#else
#define __ToPointer(a) (a)
#define __FromPointer(a) (int)(a)
#define __RmPointer(a)
#endif


#ifdef FORTRANCAPS
#define kspregisterdestroy_ KSPREGISTERDESTROY
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define kspregisterdestroy_ kspregisterdestroy
#endif
void kspregisterdestroy_(int* __ierr){
*__ierr = KSPRegisterDestroy();
}
