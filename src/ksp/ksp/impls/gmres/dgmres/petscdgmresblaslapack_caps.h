#if !defined(_DGMRESBLASLAPACK_CAPS_H)
#define _DGMRESBLASLAPACK_CAPS_H

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_SCALAR_SINGLE)
#  define LAPACKhseqr_ SHSEQR
#  define LAPACKhgeqz_ SHGEQZ
#  define LAPACKgerfs_ SGERFS
#  define LAPACKgges_ SGGES
#  define LAPACKtrsen_ STRSEN
# else
#  define LAPACKhseqr_ DHSEQR
#  define LAPACKhgeqz_ DHGEQZ
#  define LAPACKgerfs_ DGERFS
#  define LAPACKgges_ DGGES
#  define LAPACKtrsen_ DTRSEN
#  define LAPACKtgsen_ DTGSEN
# endif
# else
# if defined(PETSC_USE_SCALAR_SINGLE)
#  define LAPACKhseqr_ CHSEQR
#  define LAPACKhgeqz_ CHGEQZ
#  define LAPACKgerfs_ CGERFS
#  define LAPACKgges_ CGGES
#  define LAPACKtrsen_ CTRSEN
#  define LAPACKtgsen_ CTGSEN
# else
#  define LAPACKhseqr_ ZHSEQR
#  define LAPACKhgeqz_ ZHGEQZ
#  define LAPACKgerfs_ ZGERFS
#  define LAPACKgges_ ZGGES
#  define LAPACKtrsen_ DTRSEN
#  define LAPACKtgsen_ DTGSEN
# endif
#endif

#endif
