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
#define pcregisterdestroy_ PCREGISTERDESTROY
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define pcregisterdestroy_ pcregisterdestroy
#endif
void pcregisterdestroy_(int *__ierr){
*__ierr = PCRegisterDestroy();
}
