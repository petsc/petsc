#ifndef KNR_COMPAT_H
#define KNR_COMPAT_H 1

/* ALWAYS use prototypes */
#define USE_PROTOTYPES 1
#define USE_FN_ARGS 1

#ifdef USE_PROTOTYPES
#define Proto(x) x
#else
#define Proto(x) ()
#endif

#ifdef USE_FN_ARGS

#define ARG0(x) (void)
#define ARG1(type1,var1) (type1 var1)
#define ARG2(type1,var1,type2,var2) (type1 var1,type2 var2)
#define ARG3(type1,var1,type2,var2,type3,var3) (type1 var1,type2 var2,type3 var3)
#define ARG4(type1,var1,type2,var2,type3,var3,type4,var4) (type1 var1,type2 var2,type3 var3,type4 var4)
#define ARG5(type1,var1,type2,var2,type3,var3,type4,var4,type5,var5) (type1 var1,type2 var2,type3 var3,type4 var4,type5 var5)

#else /* Use K&R style */

#define ARG0(x) ()
#define ARG1(type1,var1) (var1) type1 var1; 
#define ARG2(type1,var1,type2,var2) (var1,var2) type1 var1; type2 var2; 
#define ARG3(type1,var1,type2,var2,type3,var3) (var1,var2,var3) type1 var1; type2 var2; type3 var3; 
#define ARG4(type1,var1,type2,var2,type3,var3,type4,var4) (var1,var2,var3,var4) type1 var1; type2 var2; type3 var3; type4 var4; 
#define ARG5(type1,var1,type2,var2,type3,var3,type4,var4,type5,var5) (var1,var2,var3,var4,var5) type1 var1; type2 var2; type3 var3; type4 var4; type5 var5; 
#endif /* K&R Decls */

#endif /* KNR_COMPAT_H */
