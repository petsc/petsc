#if !defined(_DGMRESBLASLAPACK_USCORE_H)
#define _DGMRESBLASLAPACK_USCORE_H

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_SCALAR_SINGLE)
#  define LAPACKhseqr_ shseqr_
#  define LAPACKhgeqz_ shgeqz_
#  define LAPACKgerfs_ sgerfs_
#  define LAPACKgges_ sgges_
#  define LAPACKtrsen_ strsen_
#  define LAPACKtgsen_ stgsen_
# else
#  define LAPACKhseqr_ dhseqr_
#  define LAPACKhgeqz_ dhgeqz_
#  define LAPACKgerfs_ dgerfs_
#  define LAPACKgges_ dgges_
#  define LAPACKtrsen_ dtrsen_
#  define LAPACKtgsen_ dtgsen_
# endif
# else
# if defined(PETSC_USE_SCALAR_SINGLE)
#  define LAPACKhseqr_ chseqr_
#  define LAPACKhgeqz_ chgeqz_
#  define LAPACKgerfs_ cgerfs_
#  define LAPACKgges_ cgges_
#  define LAPACKtrsen_ ctrsen_
#  define LAPACKtgsen_ ctgsen_
# else
#  define LAPACKhseqr_ zhseqr_
#  define LAPACKhgeqz_ zhgeqz_
#  define LAPACKgerfs_ zgerfs_
#  define LAPACKgges_ zgges_
#  define LAPACKtrsen_ ztrsen_
#  define LAPACKtgsen_ ztgsen_
# endif
#endif

#endif
