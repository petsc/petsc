#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/* bs = 15 for PFLOTRAN. Block operations are done by accessing all the columns   of the block at once */

PetscErrorCode MatSolve_SeqBAIJ_15_NaturalOrdering_ver2(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    n=a->mbs,*ai=a->i,*aj=a->j,*adiag=a->diag,*vi,bs=A->rmap->bs,bs2=a->bs2;
  PetscInt          i,nz,idx,idt,m;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15;
  PetscScalar       x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* forward solve the lower triangular */
  idx   = 0;
  x[0]  = b[idx];    x[1]  = b[1+idx];  x[2]  = b[2+idx];  x[3]  = b[3+idx];  x[4]  = b[4+idx];
  x[5]  = b[5+idx];  x[6]  = b[6+idx];  x[7]  = b[7+idx];  x[8]  = b[8+idx];  x[9]  = b[9+idx];
  x[10] = b[10+idx]; x[11] = b[11+idx]; x[12] = b[12+idx]; x[13] = b[13+idx]; x[14] = b[14+idx];

  for (i=1; i<n; i++) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = b[idt];    s2  = b[1+idt];  s3  = b[2+idt];  s4  = b[3+idt];  s5  = b[4+idt];
    s6  = b[5+idt];  s7  = b[6+idt];  s8  = b[7+idt];  s9  = b[8+idt];  s10 = b[9+idt];
    s11 = b[10+idt]; s12 = b[11+idt]; s13 = b[12+idt]; s14 = b[13+idt]; s15 = b[14+idt];
    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      x1  = x[idx];     x2  = x[1+idx];  x3  = x[2+idx];  x4  = x[3+idx];  x5  = x[4+idx];
      x6  = x[5+idx];   x7  = x[6+idx];  x8  = x[7+idx];  x9  = x[8+idx];  x10 = x[9+idx];
      x11 = x[10+idx]; x12  = x[11+idx]; x13 = x[12+idx]; x14 = x[13+idx]; x15 = x[14+idx];

      s1  -= v[0]*x1  + v[15]*x2 + v[30]*x3 + v[45]*x4 + v[60]*x5 + v[75]*x6 + v[90]*x7  + v[105]*x8 + v[120]*x9 + v[135]*x10 + v[150]*x11 + v[165]*x12 + v[180]*x13 + v[195]*x14 + v[210]*x15;
      s2  -= v[1]*x1  + v[16]*x2 + v[31]*x3 + v[46]*x4 + v[61]*x5 + v[76]*x6 + v[91]*x7  + v[106]*x8 + v[121]*x9 + v[136]*x10 + v[151]*x11 + v[166]*x12 + v[181]*x13 + v[196]*x14 + v[211]*x15;
      s3  -= v[2]*x1  + v[17]*x2 + v[32]*x3 + v[47]*x4 + v[62]*x5 + v[77]*x6 + v[92]*x7  + v[107]*x8 + v[122]*x9 + v[137]*x10 + v[152]*x11 + v[167]*x12 + v[182]*x13 + v[197]*x14 + v[212]*x15;
      s4  -= v[3]*x1  + v[18]*x2 + v[33]*x3 + v[48]*x4 + v[63]*x5 + v[78]*x6 + v[93]*x7  + v[108]*x8 + v[123]*x9 + v[138]*x10 + v[153]*x11 + v[168]*x12 + v[183]*x13 + v[198]*x14 + v[213]*x15;
      s5  -= v[4]*x1  + v[19]*x2 + v[34]*x3 + v[49]*x4 + v[64]*x5 + v[79]*x6 + v[94]*x7  + v[109]*x8 + v[124]*x9 + v[139]*x10 + v[154]*x11 + v[169]*x12 + v[184]*x13 + v[199]*x14 + v[214]*x15;
      s6  -= v[5]*x1  + v[20]*x2 + v[35]*x3 + v[50]*x4 + v[65]*x5 + v[80]*x6 + v[95]*x7  + v[110]*x8 + v[125]*x9 + v[140]*x10 + v[155]*x11 + v[170]*x12 + v[185]*x13 + v[200]*x14 + v[215]*x15;
      s7  -= v[6]*x1  + v[21]*x2 + v[36]*x3 + v[51]*x4 + v[66]*x5 + v[81]*x6 + v[96]*x7  + v[111]*x8 + v[126]*x9 + v[141]*x10 + v[156]*x11 + v[171]*x12 + v[186]*x13 + v[201]*x14 + v[216]*x15;
      s8  -= v[7]*x1  + v[22]*x2 + v[37]*x3 + v[52]*x4 + v[67]*x5 + v[82]*x6 + v[97]*x7  + v[112]*x8 + v[127]*x9 + v[142]*x10 + v[157]*x11 + v[172]*x12 + v[187]*x13 + v[202]*x14 + v[217]*x15;
      s9  -= v[8]*x1  + v[23]*x2 + v[38]*x3 + v[53]*x4 + v[68]*x5 + v[83]*x6 + v[98]*x7  + v[113]*x8 + v[128]*x9 + v[143]*x10 + v[158]*x11 + v[173]*x12 + v[188]*x13 + v[203]*x14 + v[218]*x15;
      s10 -= v[9]*x1  + v[24]*x2 + v[39]*x3 + v[54]*x4 + v[69]*x5 + v[84]*x6 + v[99]*x7  + v[114]*x8 + v[129]*x9 + v[144]*x10 + v[159]*x11 + v[174]*x12 + v[189]*x13 + v[204]*x14 + v[219]*x15;
      s11 -= v[10]*x1 + v[25]*x2 + v[40]*x3 + v[55]*x4 + v[70]*x5 + v[85]*x6 + v[100]*x7 + v[115]*x8 + v[130]*x9 + v[145]*x10 + v[160]*x11 + v[175]*x12 + v[190]*x13 + v[205]*x14 + v[220]*x15;
      s12 -= v[11]*x1 + v[26]*x2 + v[41]*x3 + v[56]*x4 + v[71]*x5 + v[86]*x6 + v[101]*x7 + v[116]*x8 + v[131]*x9 + v[146]*x10 + v[161]*x11 + v[176]*x12 + v[191]*x13 + v[206]*x14 + v[221]*x15;
      s13 -= v[12]*x1 + v[27]*x2 + v[42]*x3 + v[57]*x4 + v[72]*x5 + v[87]*x6 + v[102]*x7 + v[117]*x8 + v[132]*x9 + v[147]*x10 + v[162]*x11 + v[177]*x12 + v[192]*x13 + v[207]*x14 + v[222]*x15;
      s14 -= v[13]*x1 + v[28]*x2 + v[43]*x3 + v[58]*x4 + v[73]*x5 + v[88]*x6 + v[103]*x7 + v[118]*x8 + v[133]*x9 + v[148]*x10 + v[163]*x11 + v[178]*x12 + v[193]*x13 + v[208]*x14 + v[223]*x15;
      s15 -= v[14]*x1 + v[29]*x2 + v[44]*x3 + v[59]*x4 + v[74]*x5 + v[89]*x6 + v[104]*x7 + v[119]*x8 + v[134]*x9 + v[149]*x10 + v[164]*x11 + v[179]*x12 + v[194]*x13 + v[209]*x14 + v[224]*x15;

      v += bs2;
    }
    x[idt]    = s1;  x[1+idt]  = s2;  x[2+idt]  = s3;  x[3+idt]  = s4;  x[4+idt]  = s5;
    x[5+idt]  = s6;  x[6+idt]  = s7;  x[7+idt]  = s8;  x[8+idt]  = s9;  x[9+idt]  = s10;
    x[10+idt] = s11; x[11+idt] = s12; x[12+idt] = s13; x[13+idt] = s14; x[14+idt] = s15;

  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1] - 1;
    idt = bs*i;
    s1  = x[idt];     s2  = x[1+idt];  s3  = x[2+idt];  s4  = x[3+idt];  s5  = x[4+idt];
    s6  = x[5+idt];   s7  = x[6+idt];  s8  = x[7+idt];  s9  = x[8+idt];  s10 = x[9+idt];
    s11 = x[10+idt]; s12  = x[11+idt]; s13 = x[12+idt]; s14 = x[13+idt]; s15 = x[14+idt];

    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      x1  = x[idx];     x2  = x[1+idx];  x3  = x[2+idx];  x4  = x[3+idx];  x5  = x[4+idx];
      x6  = x[5+idx];   x7  = x[6+idx];  x8  = x[7+idx];  x9  = x[8+idx];  x10 = x[9+idx];
      x11 = x[10+idx]; x12  = x[11+idx]; x13 = x[12+idx]; x14 = x[13+idx]; x15 = x[14+idx];

      s1  -= v[0]*x1  + v[15]*x2 + v[30]*x3 + v[45]*x4 + v[60]*x5 + v[75]*x6 + v[90]*x7  + v[105]*x8 + v[120]*x9 + v[135]*x10 + v[150]*x11 + v[165]*x12 + v[180]*x13 + v[195]*x14 + v[210]*x15;
      s2  -= v[1]*x1  + v[16]*x2 + v[31]*x3 + v[46]*x4 + v[61]*x5 + v[76]*x6 + v[91]*x7  + v[106]*x8 + v[121]*x9 + v[136]*x10 + v[151]*x11 + v[166]*x12 + v[181]*x13 + v[196]*x14 + v[211]*x15;
      s3  -= v[2]*x1  + v[17]*x2 + v[32]*x3 + v[47]*x4 + v[62]*x5 + v[77]*x6 + v[92]*x7  + v[107]*x8 + v[122]*x9 + v[137]*x10 + v[152]*x11 + v[167]*x12 + v[182]*x13 + v[197]*x14 + v[212]*x15;
      s4  -= v[3]*x1  + v[18]*x2 + v[33]*x3 + v[48]*x4 + v[63]*x5 + v[78]*x6 + v[93]*x7  + v[108]*x8 + v[123]*x9 + v[138]*x10 + v[153]*x11 + v[168]*x12 + v[183]*x13 + v[198]*x14 + v[213]*x15;
      s5  -= v[4]*x1  + v[19]*x2 + v[34]*x3 + v[49]*x4 + v[64]*x5 + v[79]*x6 + v[94]*x7  + v[109]*x8 + v[124]*x9 + v[139]*x10 + v[154]*x11 + v[169]*x12 + v[184]*x13 + v[199]*x14 + v[214]*x15;
      s6  -= v[5]*x1  + v[20]*x2 + v[35]*x3 + v[50]*x4 + v[65]*x5 + v[80]*x6 + v[95]*x7  + v[110]*x8 + v[125]*x9 + v[140]*x10 + v[155]*x11 + v[170]*x12 + v[185]*x13 + v[200]*x14 + v[215]*x15;
      s7  -= v[6]*x1  + v[21]*x2 + v[36]*x3 + v[51]*x4 + v[66]*x5 + v[81]*x6 + v[96]*x7  + v[111]*x8 + v[126]*x9 + v[141]*x10 + v[156]*x11 + v[171]*x12 + v[186]*x13 + v[201]*x14 + v[216]*x15;
      s8  -= v[7]*x1  + v[22]*x2 + v[37]*x3 + v[52]*x4 + v[67]*x5 + v[82]*x6 + v[97]*x7  + v[112]*x8 + v[127]*x9 + v[142]*x10 + v[157]*x11 + v[172]*x12 + v[187]*x13 + v[202]*x14 + v[217]*x15;
      s9  -= v[8]*x1  + v[23]*x2 + v[38]*x3 + v[53]*x4 + v[68]*x5 + v[83]*x6 + v[98]*x7  + v[113]*x8 + v[128]*x9 + v[143]*x10 + v[158]*x11 + v[173]*x12 + v[188]*x13 + v[203]*x14 + v[218]*x15;
      s10 -= v[9]*x1  + v[24]*x2 + v[39]*x3 + v[54]*x4 + v[69]*x5 + v[84]*x6 + v[99]*x7  + v[114]*x8 + v[129]*x9 + v[144]*x10 + v[159]*x11 + v[174]*x12 + v[189]*x13 + v[204]*x14 + v[219]*x15;
      s11 -= v[10]*x1 + v[25]*x2 + v[40]*x3 + v[55]*x4 + v[70]*x5 + v[85]*x6 + v[100]*x7 + v[115]*x8 + v[130]*x9 + v[145]*x10 + v[160]*x11 + v[175]*x12 + v[190]*x13 + v[205]*x14 + v[220]*x15;
      s12 -= v[11]*x1 + v[26]*x2 + v[41]*x3 + v[56]*x4 + v[71]*x5 + v[86]*x6 + v[101]*x7 + v[116]*x8 + v[131]*x9 + v[146]*x10 + v[161]*x11 + v[176]*x12 + v[191]*x13 + v[206]*x14 + v[221]*x15;
      s13 -= v[12]*x1 + v[27]*x2 + v[42]*x3 + v[57]*x4 + v[72]*x5 + v[87]*x6 + v[102]*x7 + v[117]*x8 + v[132]*x9 + v[147]*x10 + v[162]*x11 + v[177]*x12 + v[192]*x13 + v[207]*x14 + v[222]*x15;
      s14 -= v[13]*x1 + v[28]*x2 + v[43]*x3 + v[58]*x4 + v[73]*x5 + v[88]*x6 + v[103]*x7 + v[118]*x8 + v[133]*x9 + v[148]*x10 + v[163]*x11 + v[178]*x12 + v[193]*x13 + v[208]*x14 + v[223]*x15;
      s15 -= v[14]*x1 + v[29]*x2 + v[44]*x3 + v[59]*x4 + v[74]*x5 + v[89]*x6 + v[104]*x7 + v[119]*x8 + v[134]*x9 + v[149]*x10 + v[164]*x11 + v[179]*x12 + v[194]*x13 + v[209]*x14 + v[224]*x15;

      v += bs2;
    }

    x[idt]    = v[0]*s1  + v[15]*s2 + v[30]*s3 + v[45]*s4 + v[60]*s5 + v[75]*s6 + v[90]*s7  + v[105]*s8 + v[120]*s9 + v[135]*s10 + v[150]*s11 + v[165]*s12 + v[180]*s13 + v[195]*s14 + v[210]*s15;
    x[1+idt]  = v[1]*s1  + v[16]*s2 + v[31]*s3 + v[46]*s4 + v[61]*s5 + v[76]*s6 + v[91]*s7  + v[106]*s8 + v[121]*s9 + v[136]*s10 + v[151]*s11 + v[166]*s12 + v[181]*s13 + v[196]*s14 + v[211]*s15;
    x[2+idt]  = v[2]*s1  + v[17]*s2 + v[32]*s3 + v[47]*s4 + v[62]*s5 + v[77]*s6 + v[92]*s7  + v[107]*s8 + v[122]*s9 + v[137]*s10 + v[152]*s11 + v[167]*s12 + v[182]*s13 + v[197]*s14 + v[212]*s15;
    x[3+idt]  = v[3]*s1  + v[18]*s2 + v[33]*s3 + v[48]*s4 + v[63]*s5 + v[78]*s6 + v[93]*s7  + v[108]*s8 + v[123]*s9 + v[138]*s10 + v[153]*s11 + v[168]*s12 + v[183]*s13 + v[198]*s14 + v[213]*s15;
    x[4+idt]  = v[4]*s1  + v[19]*s2 + v[34]*s3 + v[49]*s4 + v[64]*s5 + v[79]*s6 + v[94]*s7  + v[109]*s8 + v[124]*s9 + v[139]*s10 + v[154]*s11 + v[169]*s12 + v[184]*s13 + v[199]*s14 + v[214]*s15;
    x[5+idt]  = v[5]*s1  + v[20]*s2 + v[35]*s3 + v[50]*s4 + v[65]*s5 + v[80]*s6 + v[95]*s7  + v[110]*s8 + v[125]*s9 + v[140]*s10 + v[155]*s11 + v[170]*s12 + v[185]*s13 + v[200]*s14 + v[215]*s15;
    x[6+idt]  = v[6]*s1  + v[21]*s2 + v[36]*s3 + v[51]*s4 + v[66]*s5 + v[81]*s6 + v[96]*s7  + v[111]*s8 + v[126]*s9 + v[141]*s10 + v[156]*s11 + v[171]*s12 + v[186]*s13 + v[201]*s14 + v[216]*s15;
    x[7+idt]  = v[7]*s1  + v[22]*s2 + v[37]*s3 + v[52]*s4 + v[67]*s5 + v[82]*s6 + v[97]*s7  + v[112]*s8 + v[127]*s9 + v[142]*s10 + v[157]*s11 + v[172]*s12 + v[187]*s13 + v[202]*s14 + v[217]*s15;
    x[8+idt]  = v[8]*s1  + v[23]*s2 + v[38]*s3 + v[53]*s4 + v[68]*s5 + v[83]*s6 + v[98]*s7  + v[113]*s8 + v[128]*s9 + v[143]*s10 + v[158]*s11 + v[173]*s12 + v[188]*s13 + v[203]*s14 + v[218]*s15;
    x[9+idt]  = v[9]*s1  + v[24]*s2 + v[39]*s3 + v[54]*s4 + v[69]*s5 + v[84]*s6 + v[99]*s7  + v[114]*s8 + v[129]*s9 + v[144]*s10 + v[159]*s11 + v[174]*s12 + v[189]*s13 + v[204]*s14 + v[219]*s15;
    x[10+idt] = v[10]*s1 + v[25]*s2 + v[40]*s3 + v[55]*s4 + v[70]*s5 + v[85]*s6 + v[100]*s7 + v[115]*s8 + v[130]*s9 + v[145]*s10 + v[160]*s11 + v[175]*s12 + v[190]*s13 + v[205]*s14 + v[220]*s15;
    x[11+idt] = v[11]*s1 + v[26]*s2 + v[41]*s3 + v[56]*s4 + v[71]*s5 + v[86]*s6 + v[101]*s7 + v[116]*s8 + v[131]*s9 + v[146]*s10 + v[161]*s11 + v[176]*s12 + v[191]*s13 + v[206]*s14 + v[221]*s15;
    x[12+idt] = v[12]*s1 + v[27]*s2 + v[42]*s3 + v[57]*s4 + v[72]*s5 + v[87]*s6 + v[102]*s7 + v[117]*s8 + v[132]*s9 + v[147]*s10 + v[162]*s11 + v[177]*s12 + v[192]*s13 + v[207]*s14 + v[222]*s15;
    x[13+idt] = v[13]*s1 + v[28]*s2 + v[43]*s3 + v[58]*s4 + v[73]*s5 + v[88]*s6 + v[103]*s7 + v[118]*s8 + v[133]*s9 + v[148]*s10 + v[163]*s11 + v[178]*s12 + v[193]*s13 + v[208]*s14 + v[223]*s15;
    x[14+idt] = v[14]*s1 + v[29]*s2 + v[44]*s3 + v[59]*s4 + v[74]*s5 + v[89]*s6 + v[104]*s7 + v[119]*s8 + v[134]*s9 + v[149]*s10 + v[164]*s11 + v[179]*s12 + v[194]*s13 + v[209]*s14 + v[224]*s15;

  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* bs = 15 for PFLOTRAN. Block operations are done by accessing one column at at time */
/* Default MatSolve for block size 15 */

PetscErrorCode MatSolve_SeqBAIJ_15_NaturalOrdering_ver1(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    n=a->mbs,*ai=a->i,*aj=a->j,*adiag=a->diag,*vi,bs=A->rmap->bs,bs2=a->bs2;
  PetscInt          i,k,nz,idx,idt,m;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s[15];
  PetscScalar       *x,xv;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* forward solve the lower triangular */
  for (i=0; i<n; i++) {
    v         = aa + bs2*ai[i];
    vi        = aj + ai[i];
    nz        = ai[i+1] - ai[i];
    idt       = bs*i;
    x[idt]    = b[idt];    x[1+idt]  = b[1+idt];  x[2+idt]  = b[2+idt];  x[3+idt]  = b[3+idt];  x[4+idt]  = b[4+idt];
    x[5+idt]  = b[5+idt];  x[6+idt]  = b[6+idt];  x[7+idt]  = b[7+idt];  x[8+idt]  = b[8+idt];  x[9+idt] = b[9+idt];
    x[10+idt] = b[10+idt]; x[11+idt] = b[11+idt]; x[12+idt] = b[12+idt]; x[13+idt] = b[13+idt]; x[14+idt] = b[14+idt];
    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<15; k++) {
        xv         = x[k + idx];
        x[idt]    -= v[0]*xv;
        x[1+idt]  -= v[1]*xv;
        x[2+idt]  -= v[2]*xv;
        x[3+idt]  -= v[3]*xv;
        x[4+idt]  -= v[4]*xv;
        x[5+idt]  -= v[5]*xv;
        x[6+idt]  -= v[6]*xv;
        x[7+idt]  -= v[7]*xv;
        x[8+idt]  -= v[8]*xv;
        x[9+idt]  -= v[9]*xv;
        x[10+idt] -= v[10]*xv;
        x[11+idt] -= v[11]*xv;
        x[12+idt] -= v[12]*xv;
        x[13+idt] -= v[13]*xv;
        x[14+idt] -= v[14]*xv;
        v         += 15;
      }
    }
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v     = aa + bs2*(adiag[i+1]+1);
    vi    = aj + adiag[i+1]+1;
    nz    = adiag[i] - adiag[i+1] - 1;
    idt   = bs*i;
    s[0]  = x[idt];    s[1]  = x[1+idt];  s[2]  = x[2+idt];  s[3]  = x[3+idt];  s[4]  = x[4+idt];
    s[5]  = x[5+idt];  s[6]  = x[6+idt];  s[7]  = x[7+idt];  s[8]  = x[8+idt];  s[9]  = x[9+idt];
    s[10] = x[10+idt]; s[11] = x[11+idt]; s[12] = x[12+idt]; s[13] = x[13+idt]; s[14] = x[14+idt];

    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<15; k++) {
        xv     = x[k + idx];
        s[0]  -= v[0]*xv;
        s[1]  -= v[1]*xv;
        s[2]  -= v[2]*xv;
        s[3]  -= v[3]*xv;
        s[4]  -= v[4]*xv;
        s[5]  -= v[5]*xv;
        s[6]  -= v[6]*xv;
        s[7]  -= v[7]*xv;
        s[8]  -= v[8]*xv;
        s[9]  -= v[9]*xv;
        s[10] -= v[10]*xv;
        s[11] -= v[11]*xv;
        s[12] -= v[12]*xv;
        s[13] -= v[13]*xv;
        s[14] -= v[14]*xv;
        v     += 15;
      }
    }
    ierr = PetscArrayzero(x+idt,bs);CHKERRQ(ierr);
    for (k=0; k<15; k++) {
      x[idt]    += v[0]*s[k];
      x[1+idt]  += v[1]*s[k];
      x[2+idt]  += v[2]*s[k];
      x[3+idt]  += v[3]*s[k];
      x[4+idt]  += v[4]*s[k];
      x[5+idt]  += v[5]*s[k];
      x[6+idt]  += v[6]*s[k];
      x[7+idt]  += v[7]*s[k];
      x[8+idt]  += v[8]*s[k];
      x[9+idt]  += v[9]*s[k];
      x[10+idt] += v[10]*s[k];
      x[11+idt] += v[11]*s[k];
      x[12+idt] += v[12]*s[k];
      x[13+idt] += v[13]*s[k];
      x[14+idt] += v[14]*s[k];
      v         += 15;
    }
  }
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
