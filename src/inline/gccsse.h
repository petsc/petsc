/* $Id: gccsse.h,v 1.1 2001/06/20 20:42:03 buschelm Exp $ */

#if !defined (__GCC_SSE_H_)
#define __GCC_SSE_H_
PETSC_EXTERN_CXX_BEGIN

/* Might require GCC version: gcc --version =  egcs-2.91.66 or more current (earlier versions not tested) */
/* Requires GAS (as) version: as --version = GNU assembler 2.9.5 */
/* as --version = GNU assembler 2.9.1 appears to produce incorrect op code for MOVEMASK */

/* Prepare for use of SSE instructions */

#ifdef PETSC_USE_BOPT_g
typedef struct {float field[4];} _sse_register;
#  define SSE_SCOPE_BEGIN { float _aligned_stack[35]; \
                            _sse_register *XMM0,*XMM1,*XMM2,*XMM3,*XMM4,*XMM5,*XMM6,*XMM7; \
                            unsigned long _offset; _offset = (unsigned long)_aligned_stack % 16; \
                            if (_offset) _offset = (16 - _offset)/4; \
                            XMM0 = (_sse_register *)&_aligned_stack[ 0+_offset]; \
                            XMM1 = (_sse_register *)&_aligned_stack[ 4+_offset]; \
                            XMM2 = (_sse_register *)&_aligned_stack[ 8+_offset]; \
                            XMM3 = (_sse_register *)&_aligned_stack[12+_offset]; \
                            XMM4 = (_sse_register *)&_aligned_stack[16+_offset]; \
                            XMM5 = (_sse_register *)&_aligned_stack[20+_offset]; \
                            XMM6 = (_sse_register *)&_aligned_stack[24+_offset]; \
                            XMM7 = (_sse_register *)&_aligned_stack[28+_offset]; {
#  define SSE_SCOPE_END }}
#else
#  define SSE_SCOPE_BEGIN 0
#  define SSE_SCOPE_END   0
#endif

/* For use with SSE Inlined Assembly Blocks */
/* Note: SSE_ macro invokations must NOT be followed by a ; */
#ifdef PETSC_USE_BOPT_g
#  define SSE_INLINE_BEGIN_0                 SSE_INLINE_BEGIN_3(NULL,NULL,NULL)
#  define SSE_INLINE_END_0                   SSE_INLINE_END_3

#  define SSE_INLINE_BEGIN_1(arg0)           SSE_INLINE_BEGIN_3(arg0,NULL,NULL)
#  define SSE_INLINE_END_1                   SSE_INLINE_END_3

#  define SSE_INLINE_BEGIN_2(arg0,arg1)      SSE_INLINE_BEGIN_3(arg0,arg1,NULL)
#  define SSE_INLINE_END_2                   SSE_INLINE_END_3
#  define SSE_NO_OP                          "xchgl %%eax, %%eax"
#else
#  define SSE_INLINE_BEGIN_0                 __asm__ __volatile__ (
#  define SSE_INLINE_END_0                     SSE_NO_OP : : );

#  define SSE_INLINE_BEGIN_1(arg1)           { float *_tmp_arg1; _tmp_arg1=arg1; __asm__ __volatile__ (
#  define SSE_INLINE_END_1                     SSE_NO_OP : : "r"(_tmp_arg1) ); }

#  define SSE_INLINE_BEGIN_2(arg1,arg2)      { float *_tmp_arg1, *_tmp_arg2; _tmp_arg1=arg1; _tmp_arg2=arg2; \
                                               __asm__ __volatile__ (
#  define SSE_INLINE_END_2                     SSE_NO_OP : : "r"(_tmp_arg1),"r"(_tmp_arg2) ); }
#  define   SSE_NO_OP
#endif

#define SSE_INLINE_BEGIN_3(arg1,arg2,arg3)   { float *_tmp_arg1, *_tmp_arg2, *_tmp_arg3;   \
                                               _tmp_arg1=arg1; _tmp_arg2=arg2; _tmp_arg3=arg3; \
                                               __asm__ __volatile__ (
#define SSE_INLINE_END_3                       SSE_NO_OP : : "r"(_tmp_arg1),"r"(_tmp_arg2),"r"(_tmp_arg3) ); }

#define SSE_ARG_1 "0"
#define SSE_ARG_2 "1"
#define SSE_ARG_3 "2"
/* Note: If more args are to be used, be sure the debug version uses the most args allowed */

#define AV(a) __asm__ __volatile__ (a)
#define _SSE_INLINE_HALT                     : : "r"(_tmp_arg1),"r"(_tmp_arg2),"r"(_tmp_arg3) )
#define _SSE_STORE_REG(reg)                  AV(SSE_STORE_PS(SSE_ARG_1,FLOAT_0,reg) : : "r"(reg) )
#define _SSE_INLINE_RESUME                   __asm__ __volatile__ (

#ifdef PETSC_USE_BOPT_g
#  define _SSE_DEBUG_REG(reg)                  _SSE_INLINE_HALT; _SSE_STORE_REG(reg); _SSE_INLINE_RESUME
#else
#  define _SSE_DEBUG_REG(reg)
#endif

/* Offsets to use with SSE_ load/store/arithmetic memory ops */
#define FLOAT_0     "0"
#define FLOAT_1     "4"
#define FLOAT_2     "8"
#define FLOAT_3    "12"
#define FLOAT_4    "16"
#define FLOAT_5    "20"
#define FLOAT_6    "24"
#define FLOAT_7    "28"
#define FLOAT_8    "32"
#define FLOAT_9    "36"
#define FLOAT_10   "40"
#define FLOAT_11   "44"
#define FLOAT_12   "48"
#define FLOAT_13   "52"
#define FLOAT_14   "56"
#define FLOAT_15   "60"

#define FLOAT_16   "64"
#define FLOAT_24   "96"
#define FLOAT_32  "128"
#define FLOAT_40  "160"
#define FLOAT_48  "192"
#define FLOAT_56  "224"
#define FLOAT_64  "256"

#define DOUBLE_0    "0"
#define DOUBLE_1    "8"
#define DOUBLE_2   "16"
#define DOUBLE_3   "24"
#define DOUBLE_4   "32"
#define DOUBLE_5   "40"
#define DOUBLE_6   "48"
#define DOUBLE_7   "56"

#define DOUBLE_8   "64"
#define DOUBLE_16 "128"
#define DOUBLE_20 "160"
#define DOUBLE_24 "192"
#define DOUBLE_28 "224"
#define DOUBLE_32 "256"

/* The inline stubs */

/* Note: Prefetch instructions do not modify SSE registers, so O versions are g versions */
#define SSE_PREFETCH_NTA(arg,offset)      _SSE_PREFETCH_NTA(arg,offset)
#define SSE_PREFETCH_L1(arg,offset)       _SSE_PREFETCH_L1(arg,offset)
#define SSE_PREFETCH_L2(arg,offset)       _SSE_PREFETCH_L2(arg,offset)
#define SSE_PREFETCH_L3(arg,offset)       _SSE_PREFETCH_L3(arg,offset)

/* Note: Store instructions do not modify SSE registers, so O versions are g versions */
#define SSE_STORE_SS(arg,offset,srcreg)   _SSE_STORE_SS(arg,offset,srcreg)
#define SSE_STOREL_PS(arg,offset,srcreg)  _SSE_STOREL_PS(arg,offset,srcreg)
#define SSE_STOREH_PS(arg,offset,srcreg)  _SSE_STOREH_PS(arg,offset,srcreg)
#define SSE_STORE_PS(arg,offset,srcreg)   _SSE_STORE_PS(arg,offset,srcreg)
#define SSE_STOREU_PS(arg,offset,srcreg)  _SSE_STOREU_PS(arg,offset,srcreg)
#define SSE_STREAM_PS(arg,offset,srcreg)  _SSE_STREAM_PS(arg,offset,srcreg)

/* ================================================================================================ */

/* Register-Register Copy Macros */
#define SSE_COPY_SS(dstreg,srcreg)        _SSE_COPY_SS(dstreg,srcreg)        _SSE_DEBUG_REG(dstreg)  
#define SSE_COPY_PS(dstreg,srcreg)        _SSE_COPY_PS(dstreg,srcreg)        _SSE_DEBUG_REG(dstreg)

/* Load Macros */
#define SSE_LOAD_SS(arg,offset,dstreg)    _SSE_LOAD_SS(arg,offset,dstreg)    _SSE_DEBUG_REG(dstreg)
#define SSE_LOADL_PS(arg,offset,dstreg)   _SSE_LOADL_PS(arg,offset,dstreg)   _SSE_DEBUG_REG(dstreg)
#define SSE_LOADH_PS(arg,offset,dstreg)   _SSE_LOADH_PS(arg,offset,dstreg)   _SSE_DEBUG_REG(dstreg)
#define SSE_LOAD_PS(arg,offset,dstreg)    _SSE_LOAD_PS(arg,offset,dstreg)    _SSE_DEBUG_REG(dstreg)
#define SSE_LOADU_PS(arg,offset,dstreg)   _SSE_LOADU_PS(arg,offset,dstreg)   _SSE_DEBUG_REG(dstreg)

/* Shuffle */
#define SSE_SHUFFLE(dstreg,srcreg,imm)    _SSE_SHUFFLE(dstreg,srcreg,imm)    _SSE_DEBUG_REG(dstreg)

/* Multiply: A:=A*B */
#define SSE_MULT_SS(dstreg,srcreg)        _SSE_MULT_SS(dstreg,srcreg)        _SSE_DEBUG_REG(dstreg)
#define SSE_MULT_PS(dstreg,srcreg)        _SSE_MULT_PS(dstreg,srcreg)        _SSE_DEBUG_REG(dstreg)
#define SSE_MULT_SS_M(dstreg,arg,offset)  _SSE_MULT_SS_M(dstreg,arg,offset)  _SSE_DEBUG_REG(dstreg)
#define SSE_MULT_PS_M(dstreg,arg,offset)  _SSE_MULT_PS_M(dstreg,arg,offset)  _SSE_DEBUG_REG(dstreg)

/* Divide: A:=A/B */
#define SSE_DIV_SS(dstreg,srcreg)         _SSE_DIV_SS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)
#define SSE_DIV_PS(dstreg,srcreg)         _SSE_DIV_PS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)
#define SSE_DIV_SS_M(dstreg,arg,offset)   _SSE_DIV_SS_M(dstreg,arg,offset)   _SSE_DEBUG_REG(dstreg)
#define SSE_DIV_PS_M(dstreg,arg,offset)   _SSE_DIV_PS_M(dstreg,arg,offset)   _SSE_DEBUG_REG(dstreg)

/* Reciprocal: A:=1/B */
#define SSE_RECIP_SS(dstreg,srcreg)       _SSE_RECIP_SS(dstreg,srcreg)       _SSE_DEBUG_REG(dstreg)
#define SSE_RECIP_PS(dstreg,srcreg)       _SSE_RECIP_PS(dstreg,srcreg)       _SSE_DEBUG_REG(dstreg)
#define SSE_RECIP_SS_M(dstreg,arg,offset) _SSE_RECIP_SS_M(dstreg,arg,offset) _SSE_DEBUG_REG(dstreg)
#define SSE_RECIP_PS_M(dstreg,arg,offset) _SSE_RECIP_PS_M(dstreg,arg,offset) _SSE_DEBUG_REG(dstreg)

/* Add: A:=A+B */
#define SSE_ADD_SS(dstreg,srcreg)         _SSE_ADD_SS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)
#define SSE_ADD_PS(dstreg,srcreg)         _SSE_ADD_PS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)
#define SSE_ADD_SS_M(dstreg,arg,offset)   _SSE_ADD_SS_M(dstreg,arg,offset)   _SSE_DEBUG_REG(dstreg)
#define SSE_ADD_PS_M(dstreg,arg,offset)   _SSE_ADD_PS_M(dstreg,arg,offset)   _SSE_DEBUG_REG(dstreg)

/* Subtract: A:=A-B */
#define SSE_SUB_SS(dstreg,srcreg)         _SSE_SUB_SS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)
#define SSE_SUB_PS(dstreg,srcreg)         _SSE_SUB_PS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)
#define SSE_SUB_SS_M(dstreg,arg,offset)   _SSE_SUB_SS_M(dstreg,arg,offset)   _SSE_DEBUG_REG(dstreg)
#define SSE_SUB_PS_M(dstreg,arg,offset)   _SSE_SUB_PS_M(dstreg,arg,offset)   _SSE_DEBUG_REG(dstreg)

/* Logical: A:=A<op>B */
#define SSE_AND_SS(dstreg,srcreg)         _SSE_AND_SS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)
#define SSE_ANDNOT_SS(dstreg,srcreg)      _SSE_ANDNOT_SS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)
#define SSE_OR_SS(dstreg,srcreg)          _SSE_OR_SS(dstreg,srcreg)          _SSE_DEBUG_REG(dstreg)
#define SSE_XOR_SS(dstreg,srcreg)         _SSE_XOR_SS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)

#define SSE_AND_PS(dstreg,srcreg)         _SSE_AND_PS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)
#define SSE_ANDNOT_PS(dstreg,srcreg)      _SSE_ANDNOT_PS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)
#define SSE_OR_PS(dstreg,srcreg)          _SSE_OR_PS(dstreg,srcreg)          _SSE_DEBUG_REG(dstreg)
#define SSE_XOR_PS(dstreg,srcreg)         _SSE_XOR_PS(dstreg,srcreg)         _SSE_DEBUG_REG(dstreg)

/* Comparisons A:=A<compare>B */
#define SSE_CMPEQ_SS(dstreg,srcreg)       _SSE_CMPEQ_SS(dstreg,srcreg)       _SSE_DEBUG_REG(dstreg)
#define SSE_CMPLT_SS(dstreg,srcreg)       _SSE_CMPLT_SS(dstreg,srcreg)       _SSE_DEBUG_REG(dstreg)
#define SSE_CMPLE_SS(dstreg,srcreg)       _SSE_CMPLE_SS(dstreg,srcreg)       _SSE_DEBUG_REG(dstreg)
#define SSE_CMPUNORD_SS(dstreg,srcreg)    _SSE_CMPUNORD_SS(dstreg,srcreg)    _SSE_DEBUG_REG(dstreg)
#define SSE_CMPNEQ_SS(dstreg,srcreg)      _SSE_CMPNEQ_SS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)
#define SSE_CMPNLT_SS(dstreg,srcreg)      _SSE_CMPNLT_SS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)
#define SSE_CMPNLE_SS(dstreg,srcreg)      _SSE_CMPNLE_SS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)
#define SSE_CMPORD_SS(dstreg,srcreg)      _SSE_CMPORD_SS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)

#define SSE_CMPEQ_PS(dstreg,srcreg)       _SSE_CMPEQ_PS(dstreg,srcreg)       _SSE_DEBUG_REG(dstreg)
#define SSE_CMPLT_PS(dstreg,srcreg)       _SSE_CMPLT_PS(dstreg,srcreg)       _SSE_DEBUG_REG(dstreg)
#define SSE_CMPLE_PS(dstreg,srcreg)       _SSE_CMPLE_PS(dstreg,srcreg)       _SSE_DEBUG_REG(dstreg)
#define SSE_CMPUNORD_PS(dstreg,srcreg)    _SSE_CMPUNORD_PS(dstreg,srcreg)    _SSE_DEBUG_REG(dstreg)
#define SSE_CMPNEQ_PS(dstreg,srcreg)      _SSE_CMPNEQ_PS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)
#define SSE_CMPNLT_PS(dstreg,srcreg)      _SSE_CMPNLT_PS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)
#define SSE_CMPNLE_PS(dstreg,srcreg)      _SSE_CMPNLE_PS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)
#define SSE_CMPORD_PS(dstreg,srcreg)      _SSE_CMPORD_PS(dstreg,srcreg)      _SSE_DEBUG_REG(dstreg)

/* ================================================================================================ */

/* Other useful macros whose destinations are not SSE registers */

/* Movemask (for use after comparisons) */
/* Reduces 128 bit mask to an integer based on most significant bits of 32 bit parts. */
/* Since Movemask stores to integer, for type safety, this instruction must stand-alone */
/* Also, Movemask does not modify SSE registers, so debug version is trivial */ 
#define MOVEMASK(integ,srcreg)      { int _tmp_int = integ; \
                                      AV("movmskps %%" #srcreg", %0" : "=r" (_tmp_int) : : "memory"); \
                                      integ = _tmp_int; }

/* Double_4/Float_4 Conversions FPU ops because gcc on chiba is lame (NON-SSE OPS) */
/* Perhaps these should be modified to Double_2/Float_2 Conversions */
#define CONVERT_FLOAT4_DOUBLE4(dst,src)      { float *_tmp_float_ptr; double *_tmp_double_ptr; \
                                              _tmp_float_ptr = src; _tmp_double_ptr = dst; \
                                              __asm__ __volatile__ ( \
                                                "flds    (%0)\n\t" \
                                                "fstpl   (%1)\n\t" \
                                                "flds   4(%0)\n\t" \
                                                "fstpl  8(%1)\n\t" \
                                                "flds   8(%0)\n\t" \
                                                "fstpl 16(%1)\n\t" \
                                                "flds  12(%0)\n\t" \
                                                "fstpl 24(%1)" : : \
                                                "r"(_tmp_float_ptr),"r"(_tmp_double_ptr) ); }
#define CONVERT_DOUBLE4_FLOAT4(dst,src)      { float *_tmp_float_ptr; double *_tmp_double_ptr; \
                                               _tmp_float_ptr = dst; _tmp_double_ptr = src; \
                                               __asm__ __volatile__ ( \
                                                 "fldl    (%0)\n\t" \
                                                 "fstps   (%1)\n\t" \
                                                 "fldl   8(%0)\n\t" \
                                                 "fstps  4(%1)\n\t" \
                                                 "fldl  16(%0)\n\t" \
                                                 "fstps  8(%1)\n\t" \
                                                 "fldl  24(%0)\n\t" \
                                                 "fstps 12(%1)" : : \
                                                 "r"(_tmp_double_ptr),"r"(_tmp_float_ptr) ); }

/* Aligned Malloc */
#include <malloc.h>
#define SSE_MALLOC(var,size) { void *_tmp_void_ptr = *var; size_t _tmp_size; _tmp_size = size; \
                              *var = memalign(16,size); }                              
#define SSE_FREE(var)        { void *_tmp_void_ptr = var; \
                              free(var); }

/* CPUID Instruction Macros */
#define CPUID_VENDOR   "0"
#define CPUID_FEATURES "1"
#define CPUID_CACHE    "2"

#define CPUID(imm,_eax,_ebx,_ecx,_edx) { char *_tmp_imm; \
  unsigned long _tmp_eax, _tmp_ebx, _tmp_ecx, _tmp_edx; \
  _tmp_eax=*_eax; _tmp_ebx=*_ebx; _tmp_ecx=*_ecx; _tmp_edx=*_edx; \
  _tmp_imm=imm; \
  __asm__ __volatile__ ( \
    "movl $" imm ", %%eax\n\t" \
    "cpuid \n\t" \
    : "=a"(_tmp_eax),"=b"(_tmp_ebx),"=c"(_tmp_ecx),"=d"(_tmp_edx) \
    : : "memory"); \
  *_eax=_tmp_eax; *_ebx=_tmp_ebx; *_ecx=_tmp_ecx; *_edx=_tmp_edx; \
}

#define CPUID_GET_VENDOR(result) { char _gv_vendor_string[13]="************"; int _gv_i; \
  unsigned long _gv_eax=0;unsigned long _gv_ebx=0;unsigned long _gv_ecx=0;unsigned long _gv_edx=0;\
  CPUID(CPUID_VENDOR,&_gv_eax,&_gv_ebx,&_gv_ecx,&_gv_edx); \
  for (_gv_i=0;_gv_i<4;_gv_i++) _gv_vendor_string[_gv_i+0]=*(((char *)(&_gv_ebx))+_gv_i); \
  for (_gv_i=0;_gv_i<4;_gv_i++) _gv_vendor_string[_gv_i+4]=*(((char *)(&_gv_edx))+_gv_i); \
  for (_gv_i=0;_gv_i<4;_gv_i++) _gv_vendor_string[_gv_i+8]=*(((char *)(&_gv_ecx))+_gv_i); \
}

/* ================================================================================================ */
/* The actual SSE macros */

/* Prefetch Macros */
#define _SSE_PREFETCH_NTA(arg,offset)      "prefetchnta "offset"(%"arg")\n\t"
#define _SSE_PREFETCH_L1(arg,offset)       "prefetcht0  "offset"(%"arg")\n\t"
#define _SSE_PREFETCH_L2(arg,offset)       "prefetcht1  "offset"(%"arg")\n\t"
#define _SSE_PREFETCH_L3(arg,offset)       "prefetcht2  "offset"(%"arg")\n\t"

/* Store Macros */
#define _SSE_STORE_SS(arg,offset,srcreg)   "movss   %%" #srcreg", "offset"(%"arg")\n\t"
#define _SSE_STOREL_PS(arg,offset,srcreg)  "movlps  %%" #srcreg", "offset"(%"arg")\n\t"
#define _SSE_STOREH_PS(arg,offset,srcreg)  "movhps  %%" #srcreg", "offset"(%"arg")\n\t"
#define _SSE_STORE_PS(arg,offset,srcreg)   "movaps  %%" #srcreg", "offset"(%"arg")\n\t"
#define _SSE_STOREU_PS(arg,offset,srcreg)  "movups  %%" #srcreg", "offset"(%"arg")\n\t"
#define _SSE_STREAM_PS(arg,offset,srcreg)  "movntps %%" #srcreg", "offset"(%"arg")\n\t"

/* Register-Register Copy Macros */
#define _SSE_COPY_SS(dstreg,srcreg)        "movss  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_COPY_PS(dstreg,srcreg)        "movaps %%" #srcreg", %%" #dstreg"\n\t"

/* Load Macros */
#define _SSE_LOAD_SS(arg,offset,dstreg)    "movss  "offset "(%"arg"), %%" #dstreg "\n\t"
#define _SSE_LOADL_PS(arg,offset,dstreg)   "movlps "offset "(%"arg"), %%" #dstreg "\n\t"
#define _SSE_LOADH_PS(arg,offset,dstreg)   "movhps "offset "(%"arg"), %%" #dstreg "\n\t"
#define _SSE_LOAD_PS(arg,offset,dstreg)    "movaps "offset "(%"arg"), %%" #dstreg "\n\t"
#define _SSE_LOADU_PS(arg,offset,dstreg)   "movups "offset "(%"arg"), %%" #dstreg "\n\t"

/* Shuffle */
#define _SSE_SHUFFLE(dstreg,srcreg,imm)    "shufps $" #imm", %%" #srcreg", %%" #dstreg"\n\t"

/* Multiply: A:=A*B */
#define _SSE_MULT_SS(dstreg,srcreg)        "mulss  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_MULT_PS(dstreg,srcreg)        "mulps  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_MULT_SS_M(dstreg,arg,offset)  "mulss  "offset"(%"arg"), %%" #dstreg"\n\t"
#define _SSE_MULT_PS_M(dstreg,arg,offset)  "mulps  "offset"(%"arg"), %%" #dstreg"\n\t"

/* Divide: A:=A/B */
#define _SSE_DIV_SS(dstreg,srcreg)         "divss  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_DIV_PS(dstreg,srcreg)         "divps  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_DIV_SS_M(dstreg,arg,offset)   "divss  "offset"(%"arg"), %%" #dstreg"\n\t"
#define _SSE_DIV_PS_M(dstreg,arg,offset)   "divps  "offset"(%"arg"), %%" #dstreg"\n\t"

/* Reciprocal: A:=1/B */
#define _SSE_RECIP_SS(dstreg,srcreg)       "rcpss  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_RECIP_PS(dstreg,srcreg)       "rcpps  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_RECIP_SS_M(dstreg,arg,offset) "rcpss  "offset"(%"arg"), %%" #dstreg"\n\t"
#define _SSE_RECIP_PS_M(dstreg,arg,offset) "rcpps  "offset"(%"arg"), %%" #dstreg"\n\t"

/* Add: A:=A+B */
#define _SSE_ADD_SS(dstreg,srcreg)         "addss  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_ADD_PS(dstreg,srcreg)         "addps  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_ADD_SS_M(dstreg,arg,offset)   "addss  "offset"(%"arg"), %%" #dstreg"\n\t"
#define _SSE_ADD_PS_M(dstreg,arg,offset)   "addps  "offset"(%"arg"), %%" #dstreg"\n\t"

/* Subtract: A:=A-B */
#define _SSE_SUB_SS(dstreg,srcreg)         "subss  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_SUB_PS(dstreg,srcreg)         "subps  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_SUB_SS_M(dstreg,arg,offset)   "subss  "offset"(%"arg"), %%" #dstreg"\n\t"
#define _SSE_SUB_PS_M(dstreg,arg,offset)   "subps  "offset"(%"arg"), %%" #dstreg"\n\t"

/* Logical: A:=A<op>B */
#define _SSE_AND_SS(dstreg,srcreg)         "andss  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_ANDNOT_SS(dstreg,srcreg)      "andnss %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_OR_SS(dstreg,srcreg)          "orss   %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_XOR_SS(dstreg,srcreg)         "xorss  %%" #srcreg", %%" #dstreg"\n\t"

#define _SSE_AND_PS(dstreg,srcreg)         "andps  %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_ANDNOT_PS(dstreg,srcreg)      "andnps %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_OR_PS(dstreg,srcreg)          "orps   %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_XOR_PS(dstreg,srcreg)         "xorps  %%" #srcreg", %%" #dstreg"\n\t"

/* Comparisons A:=A<compare>B */
#define _SSE_CMPEQ_SS(dstreg,srcreg)       "cmpss $0, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPLT_SS(dstreg,srcreg)       "cmpss $1, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPLE_SS(dstreg,srcreg)       "cmpss $2, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPUNORD_SS(dstreg,srcreg)    "cmpss $3, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPNEQ_SS(dstreg,srcreg)      "cmpss $4, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPNLT_SS(dstreg,srcreg)      "cmpss $5, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPNLE_SS(dstreg,srcreg)      "cmpss $6, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPORD_SS(dstreg,srcreg)      "cmpss $7, %%" #srcreg", %%" #dstreg"\n\t"

#define _SSE_CMPEQ_PS(dstreg,srcreg)       "cmpps $0, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPLT_PS(dstreg,srcreg)       "cmpps $1, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPLE_PS(dstreg,srcreg)       "cmpps $2, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPUNORD_PS(dstreg,srcreg)    "cmpps $3, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPNEQ_PS(dstreg,srcreg)      "cmpps $4, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPNLT_PS(dstreg,srcreg)      "cmpps $5, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPNLE_PS(dstreg,srcreg)      "cmpps $6, %%" #srcreg", %%" #dstreg"\n\t"
#define _SSE_CMPORD_PS(dstreg,srcreg)      "cmpps $7, %%" #srcreg", %%" #dstreg"\n\t"

/* ================================================================================================ */

/* Stand Alone Versions: */

/* Note: These are only intended as a simplification of the inline form when only 1 instruction is used. */
/*       As such, the syntax is slightly different for memory versions of these operations. */
/*       Use the inline/block form in general. */

/* Prefetch Macros */
#define PREFETCH_NTA(var)        SSE_INLINE_BEGIN_1(var) SSE_PREFETCH_NTA(SSE_ARG_1,FLOAT_0)      SSE_INLINE_END_1
#define PREFETCH_L1(var)         SSE_INLINE_BEGIN_1(var) SSE_PREFETCH_L1(SSE_ARG_1,FLOAT_0)       SSE_INLINE_END_1
#define PREFETCH_L2(var)         SSE_INLINE_BEGIN_1(var) SSE_PREFETCH_L2(SSE_ARG_1,FLOAT_0)       SSE_INLINE_END_1
#define PREFETCH_L3(var)         SSE_INLINE_BEGIN_1(var) SSE_PREFETCH_L3(SSE_ARG_1,FLOAT_0)       SSE_INLINE_END_1

/* Store Macros */
#define STORE_SS(var,srcreg)     SSE_INLINE_BEGIN_1(var) SSE_STORE_SS(SSE_ARG_1,FLOAT_0,srcreg)   SSE_INLINE_END_1
#define STOREL_PS(var,srcreg)    SSE_INLINE_BEGIN_1(var) SSE_STOREL_PS(SSE_ARG_1,FLOAT_0,srcreg)  SSE_INLINE_END_1
#define STOREH_PS(var,srcreg)    SSE_INLINE_BEGIN_1(var) SSE_STOREH_PS(SSE_ARG_1,FLOAT_0,srcreg)  SSE_INLINE_END_1
#define STORE_PS(var,srcreg)     SSE_INLINE_BEGIN_1(var) SSE_STORE_PS(SSE_ARG_1,FLOAT_0,srcreg)   SSE_INLINE_END_1
#define STOREU_PS(var,srcreg)    SSE_INLINE_BEGIN_1(var) SSE_STOREU_PS(SSE_ARG_1,FLOAT_0,srcreg)  SSE_INLINE_END_1
#define STREAM_PS(var,srcreg)    SSE_INLINE_BEGIN_1(var) SSE_STREAM_PS(SSE_ARG_1,FLOAT_0,srcreg)  SSE_INLINE_END_1

/* Register-Register Copy Macros */
#define COPY_SS(dstreg,srcreg)   SSE_INLINE_BEGIN_0      SSE_COPY_SS(dstreg,srcreg)               SSE_INLINE_END_0
#define COPY_PS(dstreg,srcreg)   SSE_INLINE_BEGIN_0      SSE_COPY_PS(dstreg,srcreg)               SSE_INLINE_END_0

/* Load Macros */
#define LOAD_SS(var,dstreg)      SSE_INLINE_BEGIN_1(var) SSE_LOAD_SS(SSE_ARG_1,FLOAT_0,dstreg)    SSE_INLINE_END_1
#define LOADL_PS(var,dstreg)     SSE_INLINE_BEGIN_1(var) SSE_LOADL_PS(SSE_ARG_1,FLOAT_0,dstreg)   SSE_INLINE_END_1
#define LOADH_PS(var,dstreg)     SSE_INLINE_BEGIN_1(var) SSE_LOADH_PS(SSE_ARG_1,FLOAT_0,dstreg)   SSE_INLINE_END_1
#define LOAD_PS(var,dstreg)      SSE_INLINE_BEGIN_1(var) SSE_LOAD_PS(SSE_ARG_1,FLOAT_0,dstreg)    SSE_INLINE_END_1
#define LOADU_PS(var,dstreg)     SSE_INLINE_BEGIN_1(var) SSE_LOADU_PS(SSE_ARG_1,FLOAT_0,dstreg)   SSE_INLINE_END_1

/* Shuffle */
#define SHUFFLE(dstreg,srcreg,imm)    SSE_INLINE_BEGIN_0 SSE_SHUFFLE(dstreg,srcreg,imm)           SSE_INLINE_END_0

/* Multiply: A:=A*B */
#define MULT_SS(dstreg,srcreg)   SSE_INLINE_BEGIN_0      SSE_MULT_SS(dstreg,srcreg)               SSE_INLINE_END_0
#define MULT_PS(dstreg,srcreg)   SSE_INLINE_BEGIN_0      SSE_MULT_PS(dstreg,srcreg)               SSE_INLINE_END_0
#define MULT_SS_M(dstreg,var)    SSE_INLINE_BEGIN_1(var) SSE_MULT_SS_M(dstreg,SSE_ARG_1,FLOAT_0)  SSE_INLINE_END_1
#define MULT_PS_M(dstreg,var)    SSE_INLINE_BEGIN_1(var) SSE_MULT_PS_M(dstreg,SSE_ARG_1,FLOAT_0)  SSE_INLINE_END_1

/* Divide: A:=A/B */
#define DIV_SS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_DIV_SS(dstreg,srcreg)                SSE_INLINE_END_0
#define DIV_PS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_DIV_PS(dstreg,srcreg)                SSE_INLINE_END_0
#define DIV_SS_M(dstreg,var)     SSE_INLINE_BEGIN_1(var) SSE_DIV_SS_M(dstreg,SSE_ARG_1,FLOAT_0)   SSE_INLINE_END_1
#define DIV_PS_M(dstreg,var)     SSE_INLINE_BEGIN_1(var) SSE_DIV_PS_M(dstreg,SSE_ARG_1,FLOAT_0)   SSE_INLINE_END_1

/* Reciprocal: A:=1/B */
#define RECIP_SS(dstreg,srcreg)  SSE_INLINE_BEGIN_0      SSE_RECIP_SS(dstreg,srcreg)              SSE_INLINE_END_0
#define RECIP_PS(dstreg,srcreg)  SSE_INLINE_BEGIN_0      SSE_RECIP_PS(dstreg,srcreg)              SSE_INLINE_END_0
#define RECIP_SS_M(dstreg,var)   SSE_INLINE_BEGIN_1(var) SSE_RECIP_SS_M(dstreg,SSE_ARG_1,FLOAT_0) SSE_INLINE_END_1
#define RECIP_PS_M(dstreg,var)   SSE_INLINE_BEGIN_1(var) SSE_RECIP_PS_M(dstreg,SSE_ARG_1,FLOAT_0) SSE_INLINE_END_1

/* Add: A:=A+B */
#define ADD_SS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_ADD_SS(dstreg,srcreg)                SSE_INLINE_END_0
#define ADD_PS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_ADD_PS(dstreg,srcreg)                SSE_INLINE_END_0
#define ADD_SS_M(dstreg,var)     SSE_INLINE_BEGIN_1(var) SSE_ADD_SS_M(dstreg,SSE_ARG_1,FLOAT_0)   SSE_INLINE_END_1
#define ADD_PS_M(dstreg,var)     SSE_INLINE_BEGIN_1(var) SSE_ADD_PS_M(dstreg,SSE_ARG_1,FLOAT_0)   SSE_INLINE_END_1

/* Subtract: A:=A-B */
#define SUB_SS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_SUB_SS(dstreg,srcreg)                SSE_INLINE_END_0
#define SUB_PS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_SUB_PS(dstreg,srcreg)                SSE_INLINE_END_0
#define SUB_SS_M(dstreg,var)     SSE_INLINE_BEGIN_1(var) SSE_SUB_SS_M(dstreg,SSE_ARG_1,FLOAT_0)   SSE_INLINE_END_1
#define SUB_PS_M(dstreg,var)     SSE_INLINE_BEGIN_1(var) SSE_SUB_PS_M(dstreg,SSE_ARG_1,FLOAT_0)   SSE_INLINE_END_1

/* Note: Register - Memory versions of the following also exist, but have not been implemented */
/* Logical: A:=A<op>B */
#define AND_SS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_AND_SS(dstreg,srcreg)                SSE_INLINE_END_0
#define ANDNOT_SS(dstreg,srcreg) SSE_INLINE_BEGIN_0      SSE_ANDNOT_SS(dstreg,srcreg)             SSE_INLINE_END_0
#define OR_SS(dstreg,srcreg)     SSE_INLINE_BEGIN_0      SSE_OR_SS(dstreg,srcreg)                 SSE_INLINE_END_0
#define XOR_SS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_XOR_SS(dstreg,srcreg)                SSE_INLINE_END_0

#define AND_PS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_AND_PS(dstreg,srcreg)                SSE_INLINE_END_0
#define ANDNOT_PS(dstreg,srcreg) SSE_INLINE_BEGIN_0      SSE_ANDNOT_PS(dstreg,srcreg)             SSE_INLINE_END_0
#define OR_PS(dstreg,srcreg)     SSE_INLINE_BEGIN_0      SSE_OR_PS(dstreg,srcreg)                 SSE_INLINE_END_0
#define XOR_PS(dstreg,srcreg)    SSE_INLINE_BEGIN_0      SSE_XOR_PS(dstreg,srcreg)                SSE_INLINE_END_0

/* 
   Implementing an if():
   First perform the comparison, then use Movemask to get an integer, say i, then
   if(i) ....
*/

/* 
   Note: From the IA Software Developer's Manual:
   The greater-than relations not implemented in hardware require more than one instruction to
   emulate in software and therefore should not be implemented as pseudo-ops. (For these, the
   programmer should reverse the operands of the corresponding less than relations and use move
   instructions to ensure that the mask is moved to the correct destination register and that the
   source operand is left intact.)
*/

/* Comparisons A:=A<compare>B */
#define CMPEQ_SS(dstreg,srcreg)       SSE_INLINE_BEGIN_0 SSE_CMPEQ_SS(dstreg,srcreg)              SSE_INLINE_END_0
#define CMPLT_SS(dstreg,srcreg)       SSE_INLINE_BEGIN_0 SSE_CMPLT_SS(dstreg,srcreg)              SSE_INLINE_END_0
#define CMPLE_SS(dstreg,srcreg)       SSE_INLINE_BEGIN_0 SSE_CMPLE_SS(dstreg,srcreg)              SSE_INLINE_END_0
#define CMPUNORD_SS(dstreg,srcreg)    SSE_INLINE_BEGIN_0 SSE_CMPUNORD_SS(dstreg,srcreg)           SSE_INLINE_END_0
#define CMPNEQ_SS(dstreg,srcreg)      SSE_INLINE_BEGIN_0 SSE_CMPNEQ_SS(dstreg,srcreg)             SSE_INLINE_END_0
#define CMPNLT_SS(dstreg,srcreg)      SSE_INLINE_BEGIN_0 SSE_CMPNLT_SS(dstreg,srcreg)             SSE_INLINE_END_0
#define CMPNLE_SS(dstreg,srcreg)      SSE_INLINE_BEGIN_0 SSE_CMPNLE_SS(dstreg,srcreg)             SSE_INLINE_END_0
#define CMPORD_SS(dstreg,srcreg)      SSE_INLINE_BEGIN_0 SSE_CMPORD_SS(dstreg,srcreg)             SSE_INLINE_END_0

#define CMPEQ_PS(dstreg,srcreg)       SSE_INLINE_BEGIN_0 SSE_CMPEQ_PS(dstreg,srcreg)              SSE_INLINE_END_0
#define CMPLT_PS(dstreg,srcreg)       SSE_INLINE_BEGIN_0 SSE_CMPLT_PS(dstreg,srcreg)              SSE_INLINE_END_0
#define CMPLE_PS(dstreg,srcreg)       SSE_INLINE_BEGIN_0 SSE_CMPLE_PS(dstreg,srcreg)              SSE_INLINE_END_0
#define CMPUNORD_PS(dstreg,srcreg)    SSE_INLINE_BEGIN_0 SSE_CMPUNORD_PS(dstreg,srcreg)           SSE_INLINE_END_0
#define CMPNEQ_PS(dstreg,srcreg)      SSE_INLINE_BEGIN_0 SSE_CMPNEQ_PS(dstreg,srcreg)             SSE_INLINE_END_0
#define CMPNLT_PS(dstreg,srcreg)      SSE_INLINE_BEGIN_0 SSE_CMPNLT_PS(dstreg,srcreg)             SSE_INLINE_END_0
#define CMPNLE_PS(dstreg,srcreg)      SSE_INLINE_BEGIN_0 SSE_CMPNLE_PS(dstreg,srcreg)             SSE_INLINE_END_0
#define CMPORD_PS(dstreg,srcreg)      SSE_INLINE_BEGIN_0 SSE_CMPORD_PS(dstreg,srcreg)             SSE_INLINE_END_0

/* ================================================================================================ */

PETSC_EXTERN_CXX_END
#endif
