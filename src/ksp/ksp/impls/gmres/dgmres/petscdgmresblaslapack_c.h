#if !defined(_DGMRESBLASLAPACK_C_H)
#define _DGMRESBLASLAPACK_C_H

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_SCALAR_SINGLE)
#  define LAPACKhseqr_ shseqr
#  define LAPACKhgeqz_ shgeqz
#  define LAPACKgerfs_ sgerfs
#  define LAPACKgges_ sgges
#  define LAPACKtrsen_ strsen
#  define LAPACKtgsen_ stgsen
# else
#  define LAPACKhseqr_ dhseqr
#  define LAPACKhgeqz_ dhgeqz
#  define LAPACKgerfs_ dgerfs
#  define LAPACKgges_ dgges
#  define LAPACKtrsen_ dtrsen
#  define LAPACKtgsen_ dtgsen
# endif
# else
# if defined(PETSC_USE_SCALAR_SINGLE)
#  define LAPACKhseqr_ chseqr
#  define LAPACKhgeqz_ chgeqz
#  define LAPACKgerfs_ cgerfs
#  define LAPACKgges_ cgges
#  define LAPACKtrsen_ ctrsen
#  define LAPACKtgsen_ ctgsen
# else
#  define LAPACKhseqr_ zhseqr
#  define LAPACKhgeqz_ zhgeqz
#  define LAPACKgerfs_ zgerfs
#  define LAPACKgges_ zgges
#  define LAPACKtrsen_ ztrsen
#  define LAPACKtgsen_ ztgsen
# endif
#endif

#endif
