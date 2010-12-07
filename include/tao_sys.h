#if !defined(__TAO_SYS_H)
#define __TAO_SYS_H

#define TaoMid(a,b,c)    (((a) < (b)) ?                    \
                           (((b) < (c)) ? (b) :            \
                             (((a) < (c)) ? (c) : (a))) :  \
                           (((a) < (c)) ? (a) :            \
                             (((b) < (c)) ? (c) : (b))))

#if defined(_MSC_VER)

//Special Microsoft builtins
#define TaoNaN(a)        _isnan(a)
#define TaoInf(a)        (!_finite(a))
#define TaoInfOrNaN(a)   TaoInf(a)

#elif defined(PETSC_HAVE_ISNAN)

// These tests are from ISO C99
#define TaoNaN(a)        isnan(a)
#define TaoInf(a)        isinf(a)
#define TaoInfOrNaN(a)   (TaoInf(a) || TaoNaN(a))

#endif

#define TAO_DEFAULT        -13456834
#define TAO_INFINITY        1.0e100
#define TAO_NINFINITY       -1.0e100
#define TAO_NULL           0
#define TAO_EPSILON        PETSC_MACHINE_EPSILON


#endif


