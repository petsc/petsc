#if defined(FORTRANCAPS)
#define gen1wd_ GEN1WD
#elif !defined (FORTRANUNDERSCORE)
#define gen1wd_ gen1wd
#endif

#if defined(__cplusplus)
extern "C" {
#endif
extern void gen1wd_(int*,int*,int*,int*,int*,int*,int*,int*,int*);
#if defined(__cplusplus)
};
#endif 
#if defined(FORTRANCAPS)
#define gennd_ GENND
#elif !defined(FORTRANUNDERSCORE)
#define gennd_ gennd
#endif 

#if defined(__cplusplus)
extern "C" {
#endif
extern void gennd_(int*,int*,int*,int*,int*,int*,int*);
#if defined(__cplusplus)
};
#endif 
#if defined(FORTRANCAPS)
#define genrcm_ GENRCM
#elif !defined(FORTRANUNDERSCORE)
#define genrcm_ genrcm
#endif

#if defined(__cplusplus)
extern "C" {
#endif
extern void genrcm_(int*,int*,int*,int*,int*,int*);
#if defined(__cplusplus)
};
#endif
#if defined(FORTRANCAPS)
#define genqmd_ GENQMD 
#elif !defined(FORTRANUNDERSCORE)
#define genqmd_ genqmd
#endif

#if defined(__cplusplus)
extern "C" {
#endif
extern void genqmd_(int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*);
#if defined(__cplusplus)
};
#endif 
