
#ifndef __ICL_SSE_H_
#define __ICL_SSE_H_
#include <xmmintrin.h>
PETSC_EXTERN_CXX_BEGIN


/* SSE_FUNCTION_BEGIN must be after the LAST declaration in the outermost SSE scope */
#define SSE_SCOPE_BEGIN { __m128 XMM0,XMM1,XMM2,XMM3,XMM4,XMM5,XMM6,XMM7; {
#define SSE_SCOPE_END   }}

/* For use with SSE Inlined Assembly  Blocks */
/* Note: SSE_ macro invokations must NOT be followed by a ; */

#define SSE_INLINE_BEGIN_1(arg1)           { float *_tmp_arg1; _tmp_arg1=arg1;
#define SSE_INLINE_END_1                   }
#define SSE_INLINE_BEGIN_2(arg1,arg2)      { float *_tmp_arg1, *_tmp_arg2; _tmp_arg1=arg1; _tmp_arg2=arg2;
#define SSE_INLINE_END_2                   }
#define SSE_INLINE_BEGIN_3(arg1,arg2,arg3) { float *_tmp_arg1, *_tmp_arg2, *_tmp_arg3; \
                                             _tmp_arg1=arg1; _tmp_arg2=arg2; _tmp_arg3=arg3;
#define SSE_INLINE_END_3                   }

#define SSE_ARG_1 _tmp_arg1
#define SSE_ARG_2 _tmp_arg2
#define SSE_ARG_3 _tmp_arg3
/* Note: If more args are to be used, be sure the debug version uses the most args allowed */

/* Offset values for SSE_ load/store/arithmetic memory ops */
#define FLOAT_0    0
#define FLOAT_1    1
#define FLOAT_2    2
#define FLOAT_3    3
#define FLOAT_4    4
#define FLOAT_5    5
#define FLOAT_6    6
#define FLOAT_7    7
#define FLOAT_8    8
#define FLOAT_9    9
#define FLOAT_10  10
#define FLOAT_11  11
#define FLOAT_12  12
#define FLOAT_13  13
#define FLOAT_14  14
#define FLOAT_15  15

#define FLOAT_16  16
#define FLOAT_24  24
#define FLOAT_32  32
#define FLOAT_40  40
#define FLOAT_48  48
#define FLOAT_56  56
#define FLOAT_64  64

#define DOUBLE_0   0
#define DOUBLE_1   1
#define DOUBLE_2   2 
#define DOUBLE_3   3
#define DOUBLE_4   4
#define DOUBLE_5   5
#define DOUBLE_6   6
#define DOUBLE_7   7

#define DOUBLE_8   8
#define DOUBLE_16 16
#define DOUBLE_20 20
#define DOUBLE_24 24
#define DOUBLE_28 28
#define DOUBLE_32 32

/* xmmintrin.h provides for inline/debug versions automatically */
/* Inline versions */

/* Prefetch Macros */
#define SSE_PREFETCH_NTA(arg,offset)      PREFETCH_NTA(&arg[offset]);
#define SSE_PREFETCH_L1(arg,offset)       PREFETCH_L1(&arg[offset]);
#define SSE_PREFETCH_L2(arg,offset)       PREFETCH_L2(&arg[offset]);
#define SSE_PREFETCH_L3(arg,offset)       PREFETCH_L3(&arg[offset]);

/* Store Macros */
#define SSE_STORE_SS(arg,offset,srcreg)   STORE_SS(&arg[offset],srcreg);
#define SSE_STOREL_PS(arg,offset,srcreg)  STOREL_PS(&arg[offset],srcreg);
#define SSE_STOREH_PS(arg,offset,srcreg)  STOREH_PS(&arg[offset],srcreg);
#define SSE_STORE_PS(arg,offset,srcreg)   STORE_PS(&arg[offset],srcreg);
#define SSE_STOREU_PS(arg,offset,srcreg)  STOREU_PS(&arg[offset],srcreg);
#define SSE_STREAM_PS(arg,offset,srcreg)  STREAM_PS(&arg[offset],srcreg);

/* Register-Register Copy Macros */
#define SSE_COPY_SS(dstreg,srcreg)        COPY_SS(dstreg,srcreg);
#define SSE_COPY_PS(dstreg,srcreg)        COPY_PS(dstreg,srcreg);

/* Load Macros */
#define SSE_LOAD_SS(arg,offset,dstreg)    LOAD_SS(&arg[offset],dstreg);
#define SSE_LOADL_PS(arg,offset,dstreg)   LOADL_PS(&arg[offset],dstreg);
#define SSE_LOADH_PS(arg,offset,dstreg)   LOADH_PS(&arg[offset],dstreg);
#define SSE_LOAD_PS(arg,offset,dstreg)    LOAD_PS(&arg[offset],dstreg);
#define SSE_LOADU_PS(arg,offset,dstreg)   LOADU_PS(&arg[offset],dstreg);

/* Shuffle */
#define SSE_SHUFFLE(dstreg,srcreg,imm)    SHUFFLE(dstreg,srcreg,imm);

/* Multiply: A:=A*B */
#define SSE_MULT_SS(dstreg,srcreg)        MULT_SS(dstreg,srcreg);
#define SSE_MULT_PS(dstreg,srcreg)        MULT_PS(dstreg,srcreg);
#define SSE_MULT_SS_M(dstreg,arg,offset)  MULT_SS_M(dstreg,&arg[offset]);
#define SSE_MULT_PS_M(dstreg,arg,offset)  MULT_PS_M(dstreg,&arg[offset]);

/* Divide: A:=A/B */
#define SSE_DIV_SS(dstreg,srcreg)         DIV_SS(dstreg,srcreg);
#define SSE_DIV_PS(dstreg,srcreg)         DIV_PS(dstreg,srcreg);
#define SSE_DIV_SS_M(dstreg,arg,offset)   DIV_SS_M(dstreg,&arg[offset]);
#define SSE_DIV_PS_M(dstreg,arg,offset)   DIV_PS_M(dstreg,&arg[offset]);

/* Reciprocal: A:=1/B */
#define SSE_RECIP_SS(dstreg,srcreg)       RECIP_SS(dstreg,srcreg); 
#define SSE_RECIP_PS(dstreg,srcreg)       RECIP_PS(dstreg,srcreg);
#define SSE_RECIP_SS_M(dstreg,arg,offset) RECIP_SS_M(dstreg,&arg[offset]);
#define SSE_RECIP_PS_M(dstreg,arg,offset) RECIP_PS_M(dstreg,&arg[offset]);

/* Add: A:=A+B */
#define SSE_ADD_SS(dstreg,srcreg)         ADD_SS(dstreg,srcreg);
#define SSE_ADD_PS(dstreg,srcreg)         ADD_PS(dstreg,srcreg);
#define SSE_ADD_SS_M(dstreg,arg,offset)   ADD_SS_M(dstreg,&arg[offset]);
#define SSE_ADD_PS_M(dstreg,arg,offset)   ADD_PS_M(dstreg,&arg[offset]);

/* Subtract: A:=A-B */
#define SSE_SUB_SS(dstreg,srcreg)         SUB_SS(dstreg,srcreg);
#define SSE_SUB_PS(dstreg,srcreg)         SUB_PS(dstreg,srcreg);
#define SSE_SUB_SS_M(dstreg,arg,offset)   SUB_SS_M(dstreg,&arg[offset]);
#define SSE_SUB_PS_M(dstreg,arg,offset)   SUB_PS_M(dstreg,&arg[offset]);

/* Logical: A:=A<op>B */
#define SSE_AND_SS(dstreg,srcreg)         AND_SS(dstreg,srcreg);
#define SSE_ANDNOT_SS(dstreg,srcreg)      ANDNOT_SS(dstreg,srcreg);
#define SSE_OR_SS(dstreg,srcreg)          OR_SS(dstreg,srcreg);
#define SSE_XOR_SS(dstreg,srcreg)         XOR_SS(dstreg,srcreg);

#define SSE_AND_PS(dstreg,srcreg)         AND_PS(dstreg,srcreg);
#define SSE_ANDNOT_PS(dstreg,srcreg)      ANDNOT_PS(dstreg,srcreg);
#define SSE_OR_PS(dstreg,srcreg)          OR_PS(dstreg,srcreg);
#define SSE_XOR_PS(dstreg,srcreg)         XOR_PS(dstreg,srcreg);

/* Comparisons A:=A<compare>B */
#define SSE_CMPEQ_SS(dstreg,srcreg)       CMPEQ_SS(dstreg,srcreg);
#define SSE_CMPLT_SS(dstreg,srcreg)       CMPLT_SS(dstreg,srcreg);
#define SSE_CMPLE_SS(dstreg,srcreg)       CMPLE_SS(dstreg,srcreg);
#define SSE_CMPUNORD_SS(dstreg,srcreg)    CMPUNORD_SS(dstreg,srcreg);
#define SSE_CMPNEQ_SS(dstreg,srcreg)      CMPNEQ_SS(dstreg,srcreg);
#define SSE_CMPNLT_SS(dstreg,srcreg)      CMPNLT_SS(dstreg,srcreg);
#define SSE_CMPNLE_SS(dstreg,srcreg)      CMPNLE_SS(dstreg,srcreg);
#define SSE_CMPORD_SS(dstreg,srcreg)      CMPORD_SS(dstreg,srcreg);

#define SSE_CMPEQ_PS(dstreg,srcreg)       CMPEQ_PS(dstreg,srcreg);
#define SSE_CMPLT_PS(dstreg,srcreg)       CMPLT_PS(dstreg,srcreg);
#define SSE_CMPLE_PS(dstreg,srcreg)       CMPLE_PS(dstreg,srcreg);
#define SSE_CMPUNORD_PS(dstreg,srcreg)    CMPUNORD_PS(dstreg,srcreg);
#define SSE_CMPNEQ_PS(dstreg,srcreg)      CMPNEQ_PS(dstreg,srcreg);
#define SSE_CMPNLT_PS(dstreg,srcreg)      CMPNLT_PS(dstreg,srcreg);
#define SSE_CMPNLE_PS(dstreg,srcreg)      CMPNLE_PS(dstreg,srcreg);
#define SSE_CMPORD_PS(dstreg,srcreg)      CMPORD_PS(dstreg,srcreg);

/* ================================================================================================ */

/* Other useful macros whose destinations are not SSE registers */

/* Movemask (for use after comparisons) */
/* Reduces 128 bit mask to an integer based on most significant bits of 32 bit parts. */
#define MOVEMASK(integ,srcxmmreg)         integ = _mm_movemask_ps(srcxmmreg)

/* Double_4/Float_4 Conversions */
#define CONVERT_FLOAT4_DOUBLE4(dst,src)   { double *_tmp_double_ptr; float *_tmp_float_ptr; \
                                            _tmp_double_ptr = dst; _tmp_float_ptr = src; \
                                            _tmp_double_ptr[0]=(double)_tmp_float_ptr[0]; \
                                            _tmp_double_ptr[1]=(double)_tmp_float_ptr[1]; \
                                            _tmp_double_ptr[2]=(double)_tmp_float_ptr[2]; \
                                            _tmp_double_ptr[3]=(double)_tmp_float_ptr[3]; }

#define CONVERT_DOUBLE4_FLOAT4(dst,src)   { double *_tmp_double_ptr; float *_tmp_float_ptr; \
                                            _tmp_double_ptr = src; _tmp_float_ptr = dst; \
                                            _tmp_float_ptr[0]=(float)_tmp_double_ptr[0]; \
                                            _tmp_float_ptr[1]=(float)_tmp_double_ptr[1]; \
                                            _tmp_float_ptr[2]=(float)_tmp_double_ptr[2]; \
                                            _tmp_float_ptr[3]=(float)_tmp_double_ptr[3]; }

/* Aligned Malloc */
#define SSE_MALLOC(var,sze)              { void *_tmp_void_ptr = *var; size_t _tmp_size; _tmp_size = sze; \
                                            *var = _mm_malloc(sze,16); }
#define SSE_FREE(var)                     { void *_tmp_void_ptr = var; \
                                            _mm_free(var); }

/* CPUID Instruction Macros */

#define CPUID_VENDOR   0
#define CPUID_FEATURES 1
#define CPUID_CACHE    2

#define CPUID(imm,_eax,_ebx,_ecx,_edx) { int _tmp_imm; \
  unsigned long _tmp_eax, _tmp_ebx, _tmp_ecx, _tmp_edx; \
  _tmp_eax=*_eax; _tmp_ebx=*_ebx; _tmp_ecx=*_ecx; _tmp_edx=*_edx; \
  _tmp_imm=imm; \
  __asm { \
    __asm mov eax, imm \
    __asm cpuid \
    __asm mov _tmp_eax, eax \
    __asm mov _tmp_ebx, ebx \
    __asm mov _tmp_ecx, ecx \
    __asm mov _tmp_edx, edx \
  } \
  *_eax=_tmp_eax; *_ebx=_tmp_ebx; *_ecx=_tmp_ecx; *_edx=_tmp_edx; \
}

#define CPUID_GET_VENDOR(result) { char *_gv_vendor=result; int _gv_i; \
  unsigned long _gv_eax=0;unsigned long _gv_ebx=0;unsigned long _gv_ecx=0;unsigned long _gv_edx=0;\
  CPUID(CPUID_VENDOR,&_gv_eax,&_gv_ebx,&_gv_ecx,&_gv_edx); \
  for (_gv_i=0;_gv_i<4;_gv_i++) _gv_vendor[_gv_i+0]=*(((char *)(&_gv_ebx))+_gv_i); \
  for (_gv_i=0;_gv_i<4;_gv_i++) _gv_vendor[_gv_i+4]=*(((char *)(&_gv_edx))+_gv_i); \
  for (_gv_i=0;_gv_i<4;_gv_i++) _gv_vendor[_gv_i+8]=*(((char *)(&_gv_ecx))+_gv_i); \
}

/* ================================================================================================ */

/* The Stand Alone Versions of the SSE Macros */

/* Prefetch Macros */
#define PREFETCH_NTA(var)             _mm_prefetch((char *)(var),_MM_HINT_NTA)
#define PREFETCH_L1(var)              _mm_prefetch((char *)(var),_MM_HINT_T0)
#define PREFETCH_L2(var)              _mm_prefetch((char *)(var),_MM_HINT_T1)
#define PREFETCH_L3(var)              _mm_prefetch((char *)(var),_MM_HINT_T2)

/* Store Macros */
#define STORE_SS(var,srcreg)          _mm_store_ss(var,srcreg)
#define STOREL_PS(var,srcreg)         _mm_storel_pi((__m64 *)(var),srcreg)
#define STOREH_PS(var,srcreg)         _mm_storeh_pi((__m64 *)(var),srcreg)
#define STORE_PS(var,srcreg)          _mm_store_ps(var,srcreg)
#define STOREU_PS(var,srcreg)         _mm_storeu_ps(var,srcreg)
#define STREAM_PS(var,srcreg)         _mm_stream_ps(var,srcreg)

/* Register-Register Copy Macros */
#define COPY_SS(dstreg,srcreg)        dstreg = _mm_move_ss(dstreg,srcreg)
#define COPY_PS(dstreg,srcreg)        dstreg = srcreg

/* Load Macros */
#define LOAD_SS(var,dstreg)           dstreg = _mm_load_ss(var)
#define LOADL_PS(var,dstreg)          dstreg = _mm_loadl_pi(dstreg,(__m64 *)(var))
#define LOADH_PS(var,dstreg)          dstreg = _mm_loadh_pi(dstreg,(__m64 *)(var))
#define LOAD_PS(var,dstreg)           dstreg = _mm_load_ps(var)
#define LOADU_PS(var,dstreg)          dstreg = _mm_loadu_ps(var)

/* Shuffle */
#define SHUFFLE(dstreg,srcreg,i)      dstreg = _mm_shuffle_ps(dstreg,srcreg,i)

/* Multiply: A:=A*B */
#define MULT_SS(dstreg,srcreg)        dstreg = _mm_mul_ss(dstreg,srcreg)
#define MULT_PS(dstreg,srcreg)        dstreg = _mm_mul_ps(dstreg,srcreg)
#define MULT_SS_M(dstreg,var)         dstreg = _mm_mul_ss(dstreg,_mm_load_ss(var))
#define MULT_PS_M(dstreg,var)         dstreg = _mm_mul_ps(dstreg,_mm_load_ps(var))

/* Divide: A:=A/B */
#define DIV_SS(dstreg,srcreg)         dstreg = _mm_div_ss(dstreg,srcreg)
#define DIV_PS(dstreg,srcreg)         dstreg = _mm_div_ps(dstreg,srcreg)
#define DIV_SS_M(dstreg,var)          dstreg = _mm_div_ss(dstreg,_mm_load_ss(var))
#define DIV_PS_M(dstreg,var)          dstreg = _mm_div_ps(dstreg,_mm_load_ps(var))

/* Reciprocal: A:=1/B */
#define RECIP_SS(dstreg,srcreg)       dstreg = _mm_rcp_ss(srcreg)
#define RECIP_PS(dstreg,srcreg)       dstreg = _mm_rcp_ps(srcreg)
#define RECIP_SS_M(dstreg,var)        dstreg = _mm_rcp_ss(_mm_load_ss(var))
#define RECIP_PS_M(dstreg,var)        dstreg = _mm_rcp_ps(_mm_load_ps(var))

/* Add: A:=A+B */
#define ADD_SS(dstreg,srcreg)         dstreg = _mm_add_ss(dstreg,srcreg)
#define ADD_PS(dstreg,srcreg)         dstreg = _mm_add_ps(dstreg,srcreg)
#define ADD_SS_M(dstreg,var)          dstreg = _mm_add_ss(dstreg,_mm_load_ss(var))
#define ADD_PS_M(dstreg,var)          dstreg = _mm_add_ps(dstreg,_mm_load_ps(var))

/* Subtract: A:=A-B */
#define SUB_SS(dstreg,srcreg)         dstreg = _mm_sub_ss(dstreg,srcreg)
#define SUB_PS(dstreg,srcreg)         dstreg = _mm_sub_ps(dstreg,srcreg)
#define SUB_SS_M(dstreg,var)          dstreg = _mm_sub_ss(dstreg,_mm_load_ss(var))
#define SUB_PS_M(dstreg,var)          dstreg = _mm_sub_ps(dstreg,_mm_load_ps(var))

/* Logical: A:=A<op>B */
#define AND_SS(dstreg,srcreg)         dstreg = _mm_and_ss(dstreg,srcreg)
#define ANDNOT_SS(dstreg,srcreg)      dstreg = _mm_andnot_ss(dstreg,srcreg)
#define OR_SS(dstreg,srcreg)          dstreg = _mm_or_ss(dstreg,srcreg)
#define XOR_SS(dstreg,srcreg)         dstreg = _mm_xor_ss(dstreg,srcreg)

#define AND_PS(dstreg,srcreg)         dstreg = _mm_and_ps(dstreg,srcreg)
#define ANDNOT_PS(dstreg,srcreg)      dstreg = _mm_andnot_ps(dstreg,srcreg)
#define OR_PS(dstreg,srcreg)          dstreg = _mm_or_ps(dstreg,srcreg)
#define XOR_PS(dstreg,srcreg)         dstreg = _mm_xor_ps(dstreg,srcreg)

/* Implementing an if():
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
#define CMPEQ_SS(dstreg,srcreg)       dstreg = _mm_cmpeq_ss(dstreg,srcreg)
#define CMPLT_SS(dstreg,srcreg)       dstreg = _mm_cmplt_ss(dstreg,srcreg)
#define CMPLE_SS(dstreg,srcreg)       dstreg = _mm_cmple_ss(dstreg,srcreg)
#define CMPUNORD_SS(dstreg,srcreg)    dstreg = _mm_cmpunord_ss(dstreg,srcreg)
#define CMPNEQ_SS(dstreg,srcreg)      dstreg = _mm_cmpneq_ss(dstreg,srcreg)
#define CMPNLT_SS(dstreg,srcreg)      dstreg = _mm_cmpnlt_ss(dstreg,srcreg)
#define CMPNLE_SS(dstreg,srcreg)      dstreg = _mm_cmpnle_ss(dstreg,srcreg)
#define CMPORD_SS(dstreg,srcreg)      dstreg = _mm_cmpord_ss(dstreg,srcreg)

#define CMPEQ_PS(dstreg,srcreg)       dstreg = _mm_cmpeq_ps(dstreg,srcreg)
#define CMPLT_PS(dstreg,srcreg)       dstreg = _mm_cmplt_ps(dstreg,srcreg)
#define CMPLE_PS(dstreg,srcreg)       dstreg = _mm_cmple_ps(dstreg,srcreg)
#define CMPUNORD_PS(dstreg,srcreg)    dstreg = _mm_cmpunord_ps(dstreg,srcreg)
#define CMPNEQ_PS(dstreg,srcreg)      dstreg = _mm_cmpneq_ps(dstreg,srcreg)
#define CMPNLT_PS(dstreg,srcreg)      dstreg = _mm_cmpnlt_ps(dstreg,srcreg)
#define CMPNLE_PS(dstreg,srcreg)      dstreg = _mm_cmpnle_ps(dstreg,srcreg)
#define CMPORD_PS(dstreg,srcreg)      dstreg = _mm_cmpord_ps(dstreg,srcreg)

/* ================================================================================================ */

PETSC_EXTERN_CXX_END
#endif
