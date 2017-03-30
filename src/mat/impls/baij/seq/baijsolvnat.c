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
    ierr = PetscMemzero(x+idt,bs*sizeof(MatScalar));CHKERRQ(ierr);
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

/* Block operations are done by accessing one column at at time */
/* Default MatSolve for block size 14 */

PetscErrorCode MatSolve_SeqBAIJ_14_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    n=a->mbs,*ai=a->i,*aj=a->j,*adiag=a->diag,*vi,bs=A->rmap->bs,bs2=a->bs2;
  PetscInt          i,k,nz,idx,idt,m;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s[14];
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
    x[10+idt] = b[10+idt]; x[11+idt] = b[11+idt]; x[12+idt] = b[12+idt]; x[13+idt] = b[13+idt];
    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<bs; k++) {
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
        v         += 14;
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
    s[10] = x[10+idt]; s[11] = x[11+idt]; s[12] = x[12+idt]; s[13] = x[13+idt];

    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<bs; k++) {
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
        v     += 14;
      }
    }
    ierr = PetscMemzero(x+idt,bs*sizeof(MatScalar));CHKERRQ(ierr);
    for (k=0; k<bs; k++) {
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
      v         += 14;
    }
  }
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Block operations are done by accessing one column at at time */
/* Default MatSolve for block size 13 */

PetscErrorCode MatSolve_SeqBAIJ_13_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    n=a->mbs,*ai=a->i,*aj=a->j,*adiag=a->diag,*vi,bs=A->rmap->bs,bs2=a->bs2;
  PetscInt          i,k,nz,idx,idt,m;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s[13];
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
    x[10+idt] = b[10+idt]; x[11+idt] = b[11+idt]; x[12+idt] = b[12+idt];
    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<bs; k++) {
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
        v         += 13;
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
    s[10] = x[10+idt]; s[11] = x[11+idt]; s[12] = x[12+idt];

    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<bs; k++) {
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
        v     += 13;
      }
    }
    ierr = PetscMemzero(x+idt,bs*sizeof(MatScalar));CHKERRQ(ierr);
    for (k=0; k<bs; k++) {
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
      v         += 13;
    }
  }
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Block operations are done by accessing one column at at time */
/* Default MatSolve for block size 12 */

PetscErrorCode MatSolve_SeqBAIJ_12_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    n=a->mbs,*ai=a->i,*aj=a->j,*adiag=a->diag,*vi,bs=A->rmap->bs,bs2=a->bs2;
  PetscInt          i,k,nz,idx,idt,m;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s[12];
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
    x[10+idt] = b[10+idt]; x[11+idt] = b[11+idt];
    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<bs; k++) {
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
        v         += 12;
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
    s[10] = x[10+idt]; s[11] = x[11+idt];

    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<bs; k++) {
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
        v     += 12;
      }
    }
    ierr = PetscMemzero(x+idt,bs*sizeof(MatScalar));CHKERRQ(ierr);
    for (k=0; k<bs; k++) {
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
      v         += 12;
    }
  }
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Block operations are done by accessing one column at at time */
/* Default MatSolve for block size 11 */

PetscErrorCode MatSolve_SeqBAIJ_11_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    n=a->mbs,*ai=a->i,*aj=a->j,*adiag=a->diag,*vi,bs=A->rmap->bs,bs2=a->bs2;
  PetscInt          i,k,nz,idx,idt,m;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s[11];
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
    x[10+idt] = b[10+idt];
    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<11; k++) {
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
        v         += 11;
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
    s[10] = x[10+idt];

    for (m=0; m<nz; m++) {
      idx = bs*vi[m];
      for (k=0; k<11; k++) {
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
        v     += 11;
      }
    }
    ierr = PetscMemzero(x+idt,bs*sizeof(MatScalar));CHKERRQ(ierr);
    for (k=0; k<11; k++) {
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
      v         += 11;
    }
  }
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_7_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   = (Mat_SeqBAIJ*)A->data;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscErrorCode    ierr;
  PetscInt          i,nz,idx,idt,jdx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,s3,s4,s5,s6,s7,x1,x2,x3,x4,x5,x6,x7;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx];   x[1] = b[1+idx]; x[2] = b[2+idx];
  x[3] = b[3+idx]; x[4] = b[4+idx]; x[5] = b[5+idx];
  x[6] = b[6+idx];
  for (i=1; i<n; i++) {
    v   =  aa + 49*ai[i];
    vi  =  aj + ai[i];
    nz  =  diag[i] - ai[i];
    idx =  7*i;
    s1  =  b[idx];   s2 = b[1+idx]; s3 = b[2+idx];
    s4  =  b[3+idx]; s5 = b[4+idx]; s6 = b[5+idx];
    s7  =  b[6+idx];
    while (nz--) {
      jdx = 7*(*vi++);
      x1  = x[jdx];   x2 = x[1+jdx]; x3 = x[2+jdx];
      x4  = x[3+jdx]; x5 = x[4+jdx]; x6 = x[5+jdx];
      x7  = x[6+jdx];
      s1 -= v[0]*x1 + v[7]*x2  + v[14]*x3 + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
      s2 -= v[1]*x1 + v[8]*x2  + v[15]*x3 + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
      s3 -= v[2]*x1 + v[9]*x2  + v[16]*x3 + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
      s4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
      s5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
      s6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
      s7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v  += 49;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
    x[5+idx] = s6;
    x[6+idx] = s7;
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 49*diag[i] + 49;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    idt = 7*i;
    s1  = x[idt];   s2 = x[1+idt];
    s3  = x[2+idt]; s4 = x[3+idt];
    s5  = x[4+idt]; s6 = x[5+idt];
    s7  = x[6+idt];
    while (nz--) {
      idx = 7*(*vi++);
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];
      x4  = x[3+idx]; x5 = x[4+idx]; x6 = x[5+idx];
      x7  = x[6+idx];
      s1 -= v[0]*x1 + v[7]*x2  + v[14]*x3 + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
      s2 -= v[1]*x1 + v[8]*x2  + v[15]*x3 + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
      s3 -= v[2]*x1 + v[9]*x2  + v[16]*x3 + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
      s4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
      s5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
      s6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
      s7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v  += 49;
    }
    v      = aa + 49*diag[i];
    x[idt] = v[0]*s1 + v[7]*s2  + v[14]*s3 + v[21]*s4
             + v[28]*s5 + v[35]*s6 + v[42]*s7;
    x[1+idt] = v[1]*s1 + v[8]*s2  + v[15]*s3 + v[22]*s4
               + v[29]*s5 + v[36]*s6 + v[43]*s7;
    x[2+idt] = v[2]*s1 + v[9]*s2  + v[16]*s3 + v[23]*s4
               + v[30]*s5 + v[37]*s6 + v[44]*s7;
    x[3+idt] = v[3]*s1 + v[10]*s2  + v[17]*s3 + v[24]*s4
               + v[31]*s5 + v[38]*s6 + v[45]*s7;
    x[4+idt] = v[4]*s1 + v[11]*s2  + v[18]*s3 + v[25]*s4
               + v[32]*s5 + v[39]*s6 + v[46]*s7;
    x[5+idt] = v[5]*s1 + v[12]*s2  + v[19]*s3 + v[26]*s4
               + v[33]*s5 + v[40]*s6 + v[47]*s7;
    x[6+idt] = v[6]*s1 + v[13]*s2  + v[20]*s3 + v[27]*s4
               + v[34]*s5 + v[41]*s6 + v[48]*s7;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*36*(a->nz) - 6.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_7_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*ai=a->i,*aj=a->j,*adiag=a->diag;
  PetscErrorCode    ierr;
  PetscInt          i,k,nz,idx,jdx,idt;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,s4,s5,s6,s7,x1,x2,x3,x4,x5,x6,x7;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];x[2] = b[2+idx];x[3] = b[3+idx];
  x[4] = b[4+idx];x[5] = b[5+idx];x[6] = b[6+idx];
  for (i=1; i<n; i++) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = bs*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];
    s5  = b[4+idx];s6 = b[5+idx];s7 = b[6+idx];
    for (k=0; k<nz; k++) {
      jdx = bs*vi[k];
      x1  = x[jdx];x2 = x[1+jdx]; x3 =x[2+jdx];x4 =x[3+jdx];
      x5  = x[4+jdx]; x6 = x[5+jdx];x7 = x[6+jdx];
      s1 -= v[0]*x1 + v[7]*x2 + v[14]*x3 + v[21]*x4  + v[28]*x5 + v[35]*x6 + v[42]*x7;
      s2 -= v[1]*x1 + v[8]*x2 + v[15]*x3 + v[22]*x4  + v[29]*x5 + v[36]*x6 + v[43]*x7;
      s3 -= v[2]*x1 + v[9]*x2 + v[16]*x3 + v[23]*x4  + v[30]*x5 + v[37]*x6 + v[44]*x7;
      s4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4  + v[31]*x5 + v[38]*x6 + v[45]*x7;
      s5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4  + v[32]*x5 + v[39]*x6 + v[46]*x7;
      s6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4  + v[33]*x5 + v[40]*x6 + v[47]*x7;
      s7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4  + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v  +=  bs2;
    }

    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
    x[5+idx] = s6;
    x[6+idx] = s7;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];s3 = x[2+idt];s4 = x[3+idt];
    s5  = x[4+idt];s6 = x[5+idt];s7 = x[6+idt];
    for (k=0; k<nz; k++) {
      idx = bs*vi[k];
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];x4 = x[3+idx];
      x5  = x[4+idx];x6 = x[5+idx];x7 = x[6+idx];
      s1 -= v[0]*x1 + v[7]*x2 + v[14]*x3 + v[21]*x4  + v[28]*x5 + v[35]*x6 + v[42]*x7;
      s2 -= v[1]*x1 + v[8]*x2 + v[15]*x3 + v[22]*x4  + v[29]*x5 + v[36]*x6 + v[43]*x7;
      s3 -= v[2]*x1 + v[9]*x2 + v[16]*x3 + v[23]*x4  + v[30]*x5 + v[37]*x6 + v[44]*x7;
      s4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4  + v[31]*x5 + v[38]*x6 + v[45]*x7;
      s5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4  + v[32]*x5 + v[39]*x6 + v[46]*x7;
      s6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4  + v[33]*x5 + v[40]*x6 + v[47]*x7;
      s7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4  + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v  +=  bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[7]*s2 + v[14]*s3 + v[21]*s4  + v[28]*s5 + v[35]*s6 + v[42]*s7;
    x[1+idt] = v[1]*s1 + v[8]*s2 + v[15]*s3 + v[22]*s4  + v[29]*s5 + v[36]*s6 + v[43]*s7;
    x[2+idt] = v[2]*s1 + v[9]*s2 + v[16]*s3 + v[23]*s4  + v[30]*s5 + v[37]*s6 + v[44]*s7;
    x[3+idt] = v[3]*s1 + v[10]*s2 + v[17]*s3 + v[24]*s4  + v[31]*s5 + v[38]*s6 + v[45]*s7;
    x[4+idt] = v[4]*s1 + v[11]*s2 + v[18]*s3 + v[25]*s4  + v[32]*s5 + v[39]*s6 + v[46]*s7;
    x[5+idt] = v[5]*s1 + v[12]*s2 + v[19]*s3 + v[26]*s4  + v[33]*s5 + v[40]*s6 + v[47]*s7;
    x[6+idt] = v[6]*s1 + v[13]*s2 + v[20]*s3 + v[27]*s4  + v[34]*s5 + v[41]*s6 + v[48]*s7;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_6_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscInt          i,nz,idx,idt,jdx;
  PetscErrorCode    ierr;
  const PetscInt    *diag = a->diag,*vi,n=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa   =a->a,*v;
  PetscScalar       *x,s1,s2,s3,s4,s5,s6,x1,x2,x3,x4,x5,x6;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx];   x[1] = b[1+idx]; x[2] = b[2+idx];
  x[3] = b[3+idx]; x[4] = b[4+idx]; x[5] = b[5+idx];
  for (i=1; i<n; i++) {
    v   =  aa + 36*ai[i];
    vi  =  aj + ai[i];
    nz  =  diag[i] - ai[i];
    idx =  6*i;
    s1  =  b[idx];   s2 = b[1+idx]; s3 = b[2+idx];
    s4  =  b[3+idx]; s5 = b[4+idx]; s6 = b[5+idx];
    while (nz--) {
      jdx = 6*(*vi++);
      x1  = x[jdx];   x2 = x[1+jdx]; x3 = x[2+jdx];
      x4  = x[3+jdx]; x5 = x[4+jdx]; x6 = x[5+jdx];
      s1 -= v[0]*x1 + v[6]*x2  + v[12]*x3 + v[18]*x4 + v[24]*x5 + v[30]*x6;
      s2 -= v[1]*x1 + v[7]*x2  + v[13]*x3 + v[19]*x4 + v[25]*x5 + v[31]*x6;
      s3 -= v[2]*x1 + v[8]*x2  + v[14]*x3 + v[20]*x4 + v[26]*x5 + v[32]*x6;
      s4 -= v[3]*x1 + v[9]*x2  + v[15]*x3 + v[21]*x4 + v[27]*x5 + v[33]*x6;
      s5 -= v[4]*x1 + v[10]*x2 + v[16]*x3 + v[22]*x4 + v[28]*x5 + v[34]*x6;
      s6 -= v[5]*x1 + v[11]*x2 + v[17]*x3 + v[23]*x4 + v[29]*x5 + v[35]*x6;
      v  += 36;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
    x[5+idx] = s6;
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 36*diag[i] + 36;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    idt = 6*i;
    s1  = x[idt];   s2 = x[1+idt];
    s3  = x[2+idt]; s4 = x[3+idt];
    s5  = x[4+idt]; s6 = x[5+idt];
    while (nz--) {
      idx = 6*(*vi++);
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];
      x4  = x[3+idx]; x5 = x[4+idx]; x6 = x[5+idx];
      s1 -= v[0]*x1 + v[6]*x2  + v[12]*x3 + v[18]*x4 + v[24]*x5 + v[30]*x6;
      s2 -= v[1]*x1 + v[7]*x2  + v[13]*x3 + v[19]*x4 + v[25]*x5 + v[31]*x6;
      s3 -= v[2]*x1 + v[8]*x2  + v[14]*x3 + v[20]*x4 + v[26]*x5 + v[32]*x6;
      s4 -= v[3]*x1 + v[9]*x2  + v[15]*x3 + v[21]*x4 + v[27]*x5 + v[33]*x6;
      s5 -= v[4]*x1 + v[10]*x2 + v[16]*x3 + v[22]*x4 + v[28]*x5 + v[34]*x6;
      s6 -= v[5]*x1 + v[11]*x2 + v[17]*x3 + v[23]*x4 + v[29]*x5 + v[35]*x6;
      v  += 36;
    }
    v        = aa + 36*diag[i];
    x[idt]   = v[0]*s1 + v[6]*s2  + v[12]*s3 + v[18]*s4 + v[24]*s5 + v[30]*s6;
    x[1+idt] = v[1]*s1 + v[7]*s2  + v[13]*s3 + v[19]*s4 + v[25]*s5 + v[31]*s6;
    x[2+idt] = v[2]*s1 + v[8]*s2  + v[14]*s3 + v[20]*s4 + v[26]*s5 + v[32]*s6;
    x[3+idt] = v[3]*s1 + v[9]*s2  + v[15]*s3 + v[21]*s4 + v[27]*s5 + v[33]*s6;
    x[4+idt] = v[4]*s1 + v[10]*s2 + v[16]*s3 + v[22]*s4 + v[28]*s5 + v[34]*s6;
    x[5+idt] = v[5]*s1 + v[11]*s2 + v[17]*s3 + v[23]*s4 + v[29]*s5 + v[35]*s6;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*36*(a->nz) - 6.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_6_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*ai=a->i,*aj=a->j,*adiag=a->diag;
  PetscErrorCode    ierr;
  PetscInt          i,k,nz,idx,jdx,idt;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,s4,s5,s6,x1,x2,x3,x4,x5,x6;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];x[2] = b[2+idx];x[3] = b[3+idx];
  x[4] = b[4+idx];x[5] = b[5+idx];
  for (i=1; i<n; i++) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = bs*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];
    s5  = b[4+idx];s6 = b[5+idx];
    for (k=0; k<nz; k++) {
      jdx = bs*vi[k];
      x1  = x[jdx];x2 = x[1+jdx]; x3 =x[2+jdx];x4 =x[3+jdx];
      x5  = x[4+jdx]; x6 = x[5+jdx];
      s1 -= v[0]*x1 + v[6]*x2 + v[12]*x3 + v[18]*x4  + v[24]*x5 + v[30]*x6;
      s2 -= v[1]*x1 + v[7]*x2 + v[13]*x3 + v[19]*x4  + v[25]*x5 + v[31]*x6;;
      s3 -= v[2]*x1 + v[8]*x2 + v[14]*x3 + v[20]*x4  + v[26]*x5 + v[32]*x6;
      s4 -= v[3]*x1 + v[9]*x2 + v[15]*x3 + v[21]*x4  + v[27]*x5 + v[33]*x6;
      s5 -= v[4]*x1 + v[10]*x2 + v[16]*x3 + v[22]*x4  + v[28]*x5 + v[34]*x6;
      s6 -= v[5]*x1 + v[11]*x2 + v[17]*x3 + v[23]*x4  + v[29]*x5 + v[35]*x6;
      v  +=  bs2;
    }

    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
    x[5+idx] = s6;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];s3 = x[2+idt];s4 = x[3+idt];
    s5  = x[4+idt];s6 = x[5+idt];
    for (k=0; k<nz; k++) {
      idx = bs*vi[k];
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];x4 = x[3+idx];
      x5  = x[4+idx];x6 = x[5+idx];
      s1 -= v[0]*x1 + v[6]*x2 + v[12]*x3 + v[18]*x4  + v[24]*x5 + v[30]*x6;
      s2 -= v[1]*x1 + v[7]*x2 + v[13]*x3 + v[19]*x4  + v[25]*x5 + v[31]*x6;;
      s3 -= v[2]*x1 + v[8]*x2 + v[14]*x3 + v[20]*x4  + v[26]*x5 + v[32]*x6;
      s4 -= v[3]*x1 + v[9]*x2 + v[15]*x3 + v[21]*x4  + v[27]*x5 + v[33]*x6;
      s5 -= v[4]*x1 + v[10]*x2 + v[16]*x3 + v[22]*x4  + v[28]*x5 + v[34]*x6;
      s6 -= v[5]*x1 + v[11]*x2 + v[17]*x3 + v[23]*x4  + v[29]*x5 + v[35]*x6;
      v  +=  bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[6]*s2 + v[12]*s3 + v[18]*s4 + v[24]*s5 + v[30]*s6;
    x[1+idt] = v[1]*s1 + v[7]*s2 + v[13]*s3 + v[19]*s4 + v[25]*s5 + v[31]*s6;
    x[2+idt] = v[2]*s1 + v[8]*s2 + v[14]*s3 + v[20]*s4 + v[26]*s5 + v[32]*s6;
    x[3+idt] = v[3]*s1 + v[9]*s2 + v[15]*s3 + v[21]*s4 + v[27]*s5 + v[33]*s6;
    x[4+idt] = v[4]*s1 + v[10]*s2 + v[16]*s3 + v[22]*s4 + v[28]*s5 + v[34]*s6;
    x[5+idt] = v[5]*s1 + v[11]*s2 + v[17]*s3 + v[23]*s4 + v[29]*s5 + v[35]*s6;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_5_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   = (Mat_SeqBAIJ*)A->data;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,nz,idx,idt,jdx;
  PetscErrorCode    ierr;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,s3,s4,s5,x1,x2,x3,x4,x5;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx]; x[2] = b[2+idx]; x[3] = b[3+idx];x[4] = b[4+idx];
  for (i=1; i<n; i++) {
    v   =  aa + 25*ai[i];
    vi  =  aj + ai[i];
    nz  =  diag[i] - ai[i];
    idx =  5*i;
    s1  =  b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];s5 = b[4+idx];
    while (nz--) {
      jdx = 5*(*vi++);
      x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx];x5 = x[4+jdx];
      s1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      s2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      s3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      s4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      s5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v  += 25;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 25*diag[i] + 25;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    idt = 5*i;
    s1  = x[idt];  s2 = x[1+idt];
    s3  = x[2+idt];s4 = x[3+idt]; s5 = x[4+idt];
    while (nz--) {
      idx = 5*(*vi++);
      x1  = x[idx];   x2 = x[1+idx];x3    = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
      s1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      s2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      s3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      s4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      s5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v  += 25;
    }
    v        = aa + 25*diag[i];
    x[idt]   = v[0]*s1 + v[5]*s2 + v[10]*s3  + v[15]*s4 + v[20]*s5;
    x[1+idt] = v[1]*s1 + v[6]*s2 + v[11]*s3  + v[16]*s4 + v[21]*s5;
    x[2+idt] = v[2]*s1 + v[7]*s2 + v[12]*s3  + v[17]*s4 + v[22]*s5;
    x[3+idt] = v[3]*s1 + v[8]*s2 + v[13]*s3  + v[18]*s4 + v[23]*s5;
    x[4+idt] = v[4]*s1 + v[9]*s2 + v[14]*s3  + v[19]*s4 + v[24]*s5;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*25*(a->nz) - 5.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_5_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  = a->mbs,*vi,*ai=a->i,*aj=a->j,*adiag=a->diag;
  PetscInt          i,k,nz,idx,idt,jdx;
  PetscErrorCode    ierr;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,s3,s4,s5,x1,x2,x3,x4,x5;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx]; x[2] = b[2+idx]; x[3] = b[3+idx];x[4] = b[4+idx];
  for (i=1; i<n; i++) {
    v   = aa + 25*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = 5*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];s5 = b[4+idx];
    for (k=0; k<nz; k++) {
      jdx = 5*vi[k];
      x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx];x5 = x[4+jdx];
      s1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      s2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      s3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      s4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      s5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v  += 25;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 25*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = 5*i;
    s1  = x[idt];  s2 = x[1+idt];
    s3  = x[2+idt];s4 = x[3+idt]; s5 = x[4+idt];
    for (k=0; k<nz; k++) {
      idx = 5*vi[k];
      x1  = x[idx];   x2 = x[1+idx];x3    = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
      s1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      s2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      s3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      s4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      s5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v  += 25;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[5]*s2 + v[10]*s3  + v[15]*s4 + v[20]*s5;
    x[1+idt] = v[1]*s1 + v[6]*s2 + v[11]*s3  + v[16]*s4 + v[21]*s5;
    x[2+idt] = v[2]*s1 + v[7]*s2 + v[12]*s3  + v[17]*s4 + v[22]*s5;
    x[3+idt] = v[3]*s1 + v[8]*s2 + v[13]*s3  + v[18]*s4 + v[23]*s5;
    x[4+idt] = v[4]*s1 + v[9]*s2 + v[14]*s3  + v[19]*s4 + v[24]*s5;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*25*(a->nz) - 5.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Special case where the matrix was ILU(0) factored in the natural
   ordering. This eliminates the need for the column and row permutation.
*/
PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscInt          n  =a->mbs;
  const PetscInt    *ai=a->i,*aj=a->j;
  PetscErrorCode    ierr;
  const PetscInt    *diag = a->diag;
  const MatScalar   *aa   =a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJBLAS)
  {
    static PetscScalar w[2000]; /* very BAD need to fix */
    fortransolvebaij4blas_(&n,x,ai,aj,diag,aa,b,w);
  }
#elif defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJ)
  {
    static PetscScalar w[2000]; /* very BAD need to fix */
    fortransolvebaij4_(&n,x,ai,aj,diag,aa,b,w);
  }
#elif defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJUNROLL)
  fortransolvebaij4unroll_(&n,x,ai,aj,diag,aa,b);
#else
  {
    PetscScalar     s1,s2,s3,s4,x1,x2,x3,x4;
    const MatScalar *v;
    PetscInt        jdx,idt,idx,nz,i,ai16;
    const PetscInt  *vi;

    /* forward solve the lower triangular */
    idx  = 0;
    x[0] = b[0]; x[1] = b[1]; x[2] = b[2]; x[3] = b[3];
    for (i=1; i<n; i++) {
      v    =  aa      + 16*ai[i];
      vi   =  aj      + ai[i];
      nz   =  diag[i] - ai[i];
      idx +=  4;
      s1   =  b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];
      while (nz--) {
        jdx = 4*(*vi++);
        x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx];
        s1 -= v[0]*x1 + v[4]*x2 + v[8]*x3  + v[12]*x4;
        s2 -= v[1]*x1 + v[5]*x2 + v[9]*x3  + v[13]*x4;
        s3 -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
        s4 -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;
        v  += 16;
      }
      x[idx]   = s1;
      x[1+idx] = s2;
      x[2+idx] = s3;
      x[3+idx] = s4;
    }
    /* backward solve the upper triangular */
    idt = 4*(n-1);
    for (i=n-1; i>=0; i--) {
      ai16 = 16*diag[i];
      v    = aa + ai16 + 16;
      vi   = aj + diag[i] + 1;
      nz   = ai[i+1] - diag[i] - 1;
      s1   = x[idt];  s2 = x[1+idt];
      s3   = x[2+idt];s4 = x[3+idt];
      while (nz--) {
        idx = 4*(*vi++);
        x1  = x[idx];   x2 = x[1+idx];x3    = x[2+idx]; x4 = x[3+idx];
        s1 -= v[0]*x1 + v[4]*x2 + v[8]*x3   + v[12]*x4;
        s2 -= v[1]*x1 + v[5]*x2 + v[9]*x3   + v[13]*x4;
        s3 -= v[2]*x1 + v[6]*x2 + v[10]*x3  + v[14]*x4;
        s4 -= v[3]*x1 + v[7]*x2 + v[11]*x3  + v[15]*x4;
        v  += 16;
      }
      v        = aa + ai16;
      x[idt]   = v[0]*s1 + v[4]*s2 + v[8]*s3  + v[12]*s4;
      x[1+idt] = v[1]*s1 + v[5]*s2 + v[9]*s3  + v[13]*s4;
      x[2+idt] = v[2]*s1 + v[6]*s2 + v[10]*s3 + v[14]*s4;
      x[3+idt] = v[3]*s1 + v[7]*s2 + v[11]*s3 + v[15]*s4;
      idt     -= 4;
    }
  }
#endif

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*16*(a->nz) - 4.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n=a->mbs,*vi,*ai=a->i,*aj=a->j,*adiag=a->diag;
  PetscInt          i,k,nz,idx,jdx,idt;
  PetscErrorCode    ierr;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,s4,x1,x2,x3,x4;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];x[2] = b[2+idx];x[3] = b[3+idx];
  for (i=1; i<n; i++) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = bs*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];
    for (k=0; k<nz; k++) {
      jdx = bs*vi[k];
      x1  = x[jdx];x2 = x[1+jdx]; x3 =x[2+jdx];x4 =x[3+jdx];
      s1 -= v[0]*x1 + v[4]*x2 + v[8]*x3 + v[12]*x4;
      s2 -= v[1]*x1 + v[5]*x2 + v[9]*x3 + v[13]*x4;
      s3 -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
      s4 -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;

      v +=  bs2;
    }

    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];s3 = x[2+idt];s4 = x[3+idt];

    for (k=0; k<nz; k++) {
      idx = bs*vi[k];
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];x4 = x[3+idx];
      s1 -= v[0]*x1 + v[4]*x2 + v[8]*x3 + v[12]*x4;
      s2 -= v[1]*x1 + v[5]*x2 + v[9]*x3 + v[13]*x4;
      s3 -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
      s4 -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;

      v +=  bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[4]*s2 + v[8]*s3 + v[12]*s4;
    x[1+idt] = v[1]*s1 + v[5]*s2 + v[9]*s3 + v[13]*s4;;
    x[2+idt] = v[2]*s1 + v[6]*s2 + v[10]*s3 + v[14]*s4;
    x[3+idt] = v[3]*s1 + v[7]*s2 + v[11]*s3 + v[15]*s4;

  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_Demotion(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*ai=a->i,*aj=a->j,*diag=a->diag;
  PetscErrorCode    ierr;
  const MatScalar   *aa=a->a;
  const PetscScalar *b;
  PetscScalar       *x;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  {
    MatScalar       s1,s2,s3,s4,x1,x2,x3,x4;
    const MatScalar *v;
    MatScalar       *t=(MatScalar*)x;
    PetscInt        jdx,idt,idx,nz,i,ai16;
    const PetscInt  *vi;

    /* forward solve the lower triangular */
    idx  = 0;
    t[0] = (MatScalar)b[0];
    t[1] = (MatScalar)b[1];
    t[2] = (MatScalar)b[2];
    t[3] = (MatScalar)b[3];
    for (i=1; i<n; i++) {
      v    =  aa      + 16*ai[i];
      vi   =  aj      + ai[i];
      nz   =  diag[i] - ai[i];
      idx +=  4;
      s1   = (MatScalar)b[idx];
      s2   = (MatScalar)b[1+idx];
      s3   = (MatScalar)b[2+idx];
      s4   = (MatScalar)b[3+idx];
      while (nz--) {
        jdx = 4*(*vi++);
        x1  = t[jdx];
        x2  = t[1+jdx];
        x3  = t[2+jdx];
        x4  = t[3+jdx];
        s1 -= v[0]*x1 + v[4]*x2 + v[8]*x3  + v[12]*x4;
        s2 -= v[1]*x1 + v[5]*x2 + v[9]*x3  + v[13]*x4;
        s3 -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
        s4 -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;
        v  += 16;
      }
      t[idx]   = s1;
      t[1+idx] = s2;
      t[2+idx] = s3;
      t[3+idx] = s4;
    }
    /* backward solve the upper triangular */
    idt = 4*(n-1);
    for (i=n-1; i>=0; i--) {
      ai16 = 16*diag[i];
      v    = aa + ai16 + 16;
      vi   = aj + diag[i] + 1;
      nz   = ai[i+1] - diag[i] - 1;
      s1   = t[idt];
      s2   = t[1+idt];
      s3   = t[2+idt];
      s4   = t[3+idt];
      while (nz--) {
        idx = 4*(*vi++);
        x1  = (MatScalar)x[idx];
        x2  = (MatScalar)x[1+idx];
        x3  = (MatScalar)x[2+idx];
        x4  = (MatScalar)x[3+idx];
        s1 -= v[0]*x1 + v[4]*x2 + v[8]*x3  + v[12]*x4;
        s2 -= v[1]*x1 + v[5]*x2 + v[9]*x3  + v[13]*x4;
        s3 -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
        s4 -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;
        v  += 16;
      }
      v        = aa + ai16;
      x[idt]   = (PetscScalar)(v[0]*s1 + v[4]*s2 + v[8]*s3  + v[12]*s4);
      x[1+idt] = (PetscScalar)(v[1]*s1 + v[5]*s2 + v[9]*s3  + v[13]*s4);
      x[2+idt] = (PetscScalar)(v[2]*s1 + v[6]*s2 + v[10]*s3 + v[14]*s4);
      x[3+idt] = (PetscScalar)(v[3]*s1 + v[7]*s2 + v[11]*s3 + v[15]*s4);
      idt     -= 4;
    }
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*16*(a->nz) - 4.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SSE)

#include PETSC_HAVE_SSE
PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion_usj(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  unsigned short *aj=(unsigned short*)a->j;
  PetscErrorCode ierr;
  int            *ai=a->i,n=a->mbs,*diag = a->diag;
  MatScalar      *aa=a->a;
  PetscScalar    *x,*b;

  PetscFunctionBegin;
  SSE_SCOPE_BEGIN;
  /*
     Note: This code currently uses demotion of double
     to float when performing the mixed-mode computation.
     This may not be numerically reasonable for all applications.
  */
  PREFETCH_NTA(aa+16*ai[1]);

  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  {
    /* x will first be computed in single precision then promoted inplace to double */
    MatScalar      *v,*t=(MatScalar*)x;
    int            nz,i,idt,ai16;
    unsigned int   jdx,idx;
    unsigned short *vi;
    /* Forward solve the lower triangular factor. */

    /* First block is the identity. */
    idx = 0;
    CONVERT_DOUBLE4_FLOAT4(t,b);
    v =  aa + 16*((unsigned int)ai[1]);

    for (i=1; i<n; ) {
      PREFETCH_NTA(&v[8]);
      vi   =  aj      + ai[i];
      nz   =  diag[i] - ai[i];
      idx +=  4;

      /* Demote RHS from double to float. */
      CONVERT_DOUBLE4_FLOAT4(&t[idx],&b[idx]);
      LOAD_PS(&t[idx],XMM7);

      while (nz--) {
        PREFETCH_NTA(&v[16]);
        jdx = 4*((unsigned int)(*vi++));

        /* 4x4 Matrix-Vector product with negative accumulation: */
        SSE_INLINE_BEGIN_2(&t[jdx],v)
        SSE_LOAD_PS(SSE_ARG_1,FLOAT_0,XMM6)

        /* First Column */
        SSE_COPY_PS(XMM0,XMM6)
        SSE_SHUFFLE(XMM0,XMM0,0x00)
        SSE_MULT_PS_M(XMM0,SSE_ARG_2,FLOAT_0)
        SSE_SUB_PS(XMM7,XMM0)

        /* Second Column */
        SSE_COPY_PS(XMM1,XMM6)
        SSE_SHUFFLE(XMM1,XMM1,0x55)
        SSE_MULT_PS_M(XMM1,SSE_ARG_2,FLOAT_4)
        SSE_SUB_PS(XMM7,XMM1)

        SSE_PREFETCH_NTA(SSE_ARG_2,FLOAT_24)

        /* Third Column */
        SSE_COPY_PS(XMM2,XMM6)
        SSE_SHUFFLE(XMM2,XMM2,0xAA)
        SSE_MULT_PS_M(XMM2,SSE_ARG_2,FLOAT_8)
        SSE_SUB_PS(XMM7,XMM2)

        /* Fourth Column */
        SSE_COPY_PS(XMM3,XMM6)
        SSE_SHUFFLE(XMM3,XMM3,0xFF)
        SSE_MULT_PS_M(XMM3,SSE_ARG_2,FLOAT_12)
        SSE_SUB_PS(XMM7,XMM3)
        SSE_INLINE_END_2

        v += 16;
      }
      v =  aa + 16*ai[++i];
      PREFETCH_NTA(v);
      STORE_PS(&t[idx],XMM7);
    }

    /* Backward solve the upper triangular factor.*/

    idt  = 4*(n-1);
    ai16 = 16*diag[n-1];
    v    = aa + ai16 + 16;
    for (i=n-1; i>=0; ) {
      PREFETCH_NTA(&v[8]);
      vi = aj + diag[i] + 1;
      nz = ai[i+1] - diag[i] - 1;

      LOAD_PS(&t[idt],XMM7);

      while (nz--) {
        PREFETCH_NTA(&v[16]);
        idx = 4*((unsigned int)(*vi++));

        /* 4x4 Matrix-Vector Product with negative accumulation: */
        SSE_INLINE_BEGIN_2(&t[idx],v)
        SSE_LOAD_PS(SSE_ARG_1,FLOAT_0,XMM6)

        /* First Column */
        SSE_COPY_PS(XMM0,XMM6)
        SSE_SHUFFLE(XMM0,XMM0,0x00)
        SSE_MULT_PS_M(XMM0,SSE_ARG_2,FLOAT_0)
        SSE_SUB_PS(XMM7,XMM0)

        /* Second Column */
        SSE_COPY_PS(XMM1,XMM6)
        SSE_SHUFFLE(XMM1,XMM1,0x55)
        SSE_MULT_PS_M(XMM1,SSE_ARG_2,FLOAT_4)
        SSE_SUB_PS(XMM7,XMM1)

        SSE_PREFETCH_NTA(SSE_ARG_2,FLOAT_24)

        /* Third Column */
        SSE_COPY_PS(XMM2,XMM6)
        SSE_SHUFFLE(XMM2,XMM2,0xAA)
        SSE_MULT_PS_M(XMM2,SSE_ARG_2,FLOAT_8)
        SSE_SUB_PS(XMM7,XMM2)

        /* Fourth Column */
        SSE_COPY_PS(XMM3,XMM6)
        SSE_SHUFFLE(XMM3,XMM3,0xFF)
        SSE_MULT_PS_M(XMM3,SSE_ARG_2,FLOAT_12)
        SSE_SUB_PS(XMM7,XMM3)
        SSE_INLINE_END_2
        v += 16;
      }
      v    = aa + ai16;
      ai16 = 16*diag[--i];
      PREFETCH_NTA(aa+ai16+16);
      /*
         Scale the result by the diagonal 4x4 block,
         which was inverted as part of the factorization
      */
      SSE_INLINE_BEGIN_3(v,&t[idt],aa+ai16)
      /* First Column */
      SSE_COPY_PS(XMM0,XMM7)
      SSE_SHUFFLE(XMM0,XMM0,0x00)
      SSE_MULT_PS_M(XMM0,SSE_ARG_1,FLOAT_0)

      /* Second Column */
      SSE_COPY_PS(XMM1,XMM7)
      SSE_SHUFFLE(XMM1,XMM1,0x55)
      SSE_MULT_PS_M(XMM1,SSE_ARG_1,FLOAT_4)
      SSE_ADD_PS(XMM0,XMM1)

      SSE_PREFETCH_NTA(SSE_ARG_3,FLOAT_24)

      /* Third Column */
      SSE_COPY_PS(XMM2,XMM7)
      SSE_SHUFFLE(XMM2,XMM2,0xAA)
      SSE_MULT_PS_M(XMM2,SSE_ARG_1,FLOAT_8)
      SSE_ADD_PS(XMM0,XMM2)

      /* Fourth Column */
      SSE_COPY_PS(XMM3,XMM7)
      SSE_SHUFFLE(XMM3,XMM3,0xFF)
      SSE_MULT_PS_M(XMM3,SSE_ARG_1,FLOAT_12)
      SSE_ADD_PS(XMM0,XMM3)

      SSE_STORE_PS(SSE_ARG_2,FLOAT_0,XMM0)
      SSE_INLINE_END_3

      v    = aa + ai16 + 16;
      idt -= 4;
    }

    /* Convert t from single precision back to double precision (inplace)*/
    idt = 4*(n-1);
    for (i=n-1; i>=0; i--) {
      /*     CONVERT_FLOAT4_DOUBLE4(&x[idt],&t[idt]); */
      /* Unfortunately, CONVERT_ will count from 0 to 3 which doesn't work here. */
      PetscScalar *xtemp=&x[idt];
      MatScalar   *ttemp=&t[idt];
      xtemp[3] = (PetscScalar)ttemp[3];
      xtemp[2] = (PetscScalar)ttemp[2];
      xtemp[1] = (PetscScalar)ttemp[1];
      xtemp[0] = (PetscScalar)ttemp[0];
      idt     -= 4;
    }

  } /* End of artificial scope. */
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*16*(a->nz) - 4.0*A->cmap->n);CHKERRQ(ierr);
  SSE_SCOPE_END;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data;
  int            *aj=a->j;
  PetscErrorCode ierr;
  int            *ai=a->i,n=a->mbs,*diag = a->diag;
  MatScalar      *aa=a->a;
  PetscScalar    *x,*b;

  PetscFunctionBegin;
  SSE_SCOPE_BEGIN;
  /*
     Note: This code currently uses demotion of double
     to float when performing the mixed-mode computation.
     This may not be numerically reasonable for all applications.
  */
  PREFETCH_NTA(aa+16*ai[1]);

  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  {
    /* x will first be computed in single precision then promoted inplace to double */
    MatScalar *v,*t=(MatScalar*)x;
    int       nz,i,idt,ai16;
    int       jdx,idx;
    int       *vi;
    /* Forward solve the lower triangular factor. */

    /* First block is the identity. */
    idx = 0;
    CONVERT_DOUBLE4_FLOAT4(t,b);
    v =  aa + 16*ai[1];

    for (i=1; i<n; ) {
      PREFETCH_NTA(&v[8]);
      vi   =  aj      + ai[i];
      nz   =  diag[i] - ai[i];
      idx +=  4;

      /* Demote RHS from double to float. */
      CONVERT_DOUBLE4_FLOAT4(&t[idx],&b[idx]);
      LOAD_PS(&t[idx],XMM7);

      while (nz--) {
        PREFETCH_NTA(&v[16]);
        jdx = 4*(*vi++);
/*          jdx = *vi++; */

        /* 4x4 Matrix-Vector product with negative accumulation: */
        SSE_INLINE_BEGIN_2(&t[jdx],v)
        SSE_LOAD_PS(SSE_ARG_1,FLOAT_0,XMM6)

        /* First Column */
        SSE_COPY_PS(XMM0,XMM6)
        SSE_SHUFFLE(XMM0,XMM0,0x00)
        SSE_MULT_PS_M(XMM0,SSE_ARG_2,FLOAT_0)
        SSE_SUB_PS(XMM7,XMM0)

        /* Second Column */
        SSE_COPY_PS(XMM1,XMM6)
        SSE_SHUFFLE(XMM1,XMM1,0x55)
        SSE_MULT_PS_M(XMM1,SSE_ARG_2,FLOAT_4)
        SSE_SUB_PS(XMM7,XMM1)

        SSE_PREFETCH_NTA(SSE_ARG_2,FLOAT_24)

        /* Third Column */
        SSE_COPY_PS(XMM2,XMM6)
        SSE_SHUFFLE(XMM2,XMM2,0xAA)
        SSE_MULT_PS_M(XMM2,SSE_ARG_2,FLOAT_8)
        SSE_SUB_PS(XMM7,XMM2)

        /* Fourth Column */
        SSE_COPY_PS(XMM3,XMM6)
        SSE_SHUFFLE(XMM3,XMM3,0xFF)
        SSE_MULT_PS_M(XMM3,SSE_ARG_2,FLOAT_12)
        SSE_SUB_PS(XMM7,XMM3)
        SSE_INLINE_END_2

        v += 16;
      }
      v =  aa + 16*ai[++i];
      PREFETCH_NTA(v);
      STORE_PS(&t[idx],XMM7);
    }

    /* Backward solve the upper triangular factor.*/

    idt  = 4*(n-1);
    ai16 = 16*diag[n-1];
    v    = aa + ai16 + 16;
    for (i=n-1; i>=0; ) {
      PREFETCH_NTA(&v[8]);
      vi = aj + diag[i] + 1;
      nz = ai[i+1] - diag[i] - 1;

      LOAD_PS(&t[idt],XMM7);

      while (nz--) {
        PREFETCH_NTA(&v[16]);
        idx = 4*(*vi++);
/*          idx = *vi++; */

        /* 4x4 Matrix-Vector Product with negative accumulation: */
        SSE_INLINE_BEGIN_2(&t[idx],v)
        SSE_LOAD_PS(SSE_ARG_1,FLOAT_0,XMM6)

        /* First Column */
        SSE_COPY_PS(XMM0,XMM6)
        SSE_SHUFFLE(XMM0,XMM0,0x00)
        SSE_MULT_PS_M(XMM0,SSE_ARG_2,FLOAT_0)
        SSE_SUB_PS(XMM7,XMM0)

        /* Second Column */
        SSE_COPY_PS(XMM1,XMM6)
        SSE_SHUFFLE(XMM1,XMM1,0x55)
        SSE_MULT_PS_M(XMM1,SSE_ARG_2,FLOAT_4)
        SSE_SUB_PS(XMM7,XMM1)

        SSE_PREFETCH_NTA(SSE_ARG_2,FLOAT_24)

        /* Third Column */
        SSE_COPY_PS(XMM2,XMM6)
        SSE_SHUFFLE(XMM2,XMM2,0xAA)
        SSE_MULT_PS_M(XMM2,SSE_ARG_2,FLOAT_8)
        SSE_SUB_PS(XMM7,XMM2)

        /* Fourth Column */
        SSE_COPY_PS(XMM3,XMM6)
        SSE_SHUFFLE(XMM3,XMM3,0xFF)
        SSE_MULT_PS_M(XMM3,SSE_ARG_2,FLOAT_12)
        SSE_SUB_PS(XMM7,XMM3)
        SSE_INLINE_END_2
        v += 16;
      }
      v    = aa + ai16;
      ai16 = 16*diag[--i];
      PREFETCH_NTA(aa+ai16+16);
      /*
         Scale the result by the diagonal 4x4 block,
         which was inverted as part of the factorization
      */
      SSE_INLINE_BEGIN_3(v,&t[idt],aa+ai16)
      /* First Column */
      SSE_COPY_PS(XMM0,XMM7)
      SSE_SHUFFLE(XMM0,XMM0,0x00)
      SSE_MULT_PS_M(XMM0,SSE_ARG_1,FLOAT_0)

      /* Second Column */
      SSE_COPY_PS(XMM1,XMM7)
      SSE_SHUFFLE(XMM1,XMM1,0x55)
      SSE_MULT_PS_M(XMM1,SSE_ARG_1,FLOAT_4)
      SSE_ADD_PS(XMM0,XMM1)

      SSE_PREFETCH_NTA(SSE_ARG_3,FLOAT_24)

      /* Third Column */
      SSE_COPY_PS(XMM2,XMM7)
      SSE_SHUFFLE(XMM2,XMM2,0xAA)
      SSE_MULT_PS_M(XMM2,SSE_ARG_1,FLOAT_8)
      SSE_ADD_PS(XMM0,XMM2)

      /* Fourth Column */
      SSE_COPY_PS(XMM3,XMM7)
      SSE_SHUFFLE(XMM3,XMM3,0xFF)
      SSE_MULT_PS_M(XMM3,SSE_ARG_1,FLOAT_12)
      SSE_ADD_PS(XMM0,XMM3)

      SSE_STORE_PS(SSE_ARG_2,FLOAT_0,XMM0)
      SSE_INLINE_END_3

      v    = aa + ai16 + 16;
      idt -= 4;
    }

    /* Convert t from single precision back to double precision (inplace)*/
    idt = 4*(n-1);
    for (i=n-1; i>=0; i--) {
      /*     CONVERT_FLOAT4_DOUBLE4(&x[idt],&t[idt]); */
      /* Unfortunately, CONVERT_ will count from 0 to 3 which doesn't work here. */
      PetscScalar *xtemp=&x[idt];
      MatScalar   *ttemp=&t[idt];
      xtemp[3] = (PetscScalar)ttemp[3];
      xtemp[2] = (PetscScalar)ttemp[2];
      xtemp[1] = (PetscScalar)ttemp[1];
      xtemp[0] = (PetscScalar)ttemp[0];
      idt     -= 4;
    }

  } /* End of artificial scope. */
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*16*(a->nz) - 4.0*A->cmap->n);CHKERRQ(ierr);
  SSE_SCOPE_END;
  PetscFunctionReturn(0);
}

#endif

/*
      Special case where the matrix was ILU(0) factored in the natural
   ordering. This eliminates the need for the column and row permutation.
*/
PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*ai=a->i,*aj=a->j;
  PetscErrorCode    ierr;
  const PetscInt    *diag = a->diag,*vi;
  const MatScalar   *aa   =a->a,*v;
  PetscScalar       *x,s1,s2,s3,x1,x2,x3;
  const PetscScalar *b;
  PetscInt          jdx,idt,idx,nz,i;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[0]; x[1] = b[1]; x[2] = b[2];
  for (i=1; i<n; i++) {
    v    =  aa      + 9*ai[i];
    vi   =  aj      + ai[i];
    nz   =  diag[i] - ai[i];
    idx +=  3;
    s1   =  b[idx];s2 = b[1+idx];s3 = b[2+idx];
    while (nz--) {
      jdx = 3*(*vi++);
      x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;
      v  += 9;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 9*diag[i] + 9;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    idt = 3*i;
    s1  = x[idt];  s2 = x[1+idt];
    s3  = x[2+idt];
    while (nz--) {
      idx = 3*(*vi++);
      x1  = x[idx];   x2 = x[1+idx];x3    = x[2+idx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;
      v  += 9;
    }
    v        = aa +  9*diag[i];
    x[idt]   = v[0]*s1 + v[3]*s2 + v[6]*s3;
    x[1+idt] = v[1]*s1 + v[4]*s2 + v[7]*s3;
    x[2+idt] = v[2]*s1 + v[5]*s2 + v[8]*s3;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*9*(a->nz) - 3.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*ai=a->i,*aj=a->j,*adiag=a->diag;
  PetscErrorCode    ierr;
  PetscInt          i,k,nz,idx,jdx,idt;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,x1,x2,x3;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];x[2] = b[2+idx];
  for (i=1; i<n; i++) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = bs*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];
    for (k=0; k<nz; k++) {
      jdx = bs*vi[k];
      x1  = x[jdx];x2 = x[1+jdx]; x3 =x[2+jdx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;

      v +=  bs2;
    }

    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];s3 = x[2+idt];

    for (k=0; k<nz; k++) {
      idx = bs*vi[k];
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;

      v +=  bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[3]*s2 + v[6]*s3;
    x[1+idt] = v[1]*s1 + v[4]*s2 + v[7]*s3;
    x[2+idt] = v[2]*s1 + v[5]*s2 + v[8]*s3;

  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqBAIJ_3_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscErrorCode    ierr;
  PetscInt          i,k,nz,idx,jdx;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,x1,x2,x3;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];x[2] = b[2+idx];
  for (i=1; i<n; i++) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = bs*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];
    for (k=0; k<nz; k++) {
      jdx = bs*vi[k];
      x1  = x[jdx];x2 = x[1+jdx]; x3 =x[2+jdx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;

      v +=  bs2;
    }

    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(1.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatBackwardSolve_SeqBAIJ_3_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*aj=a->j,*adiag=a->diag;
  PetscErrorCode    ierr;
  PetscInt          i,k,nz,idx,idt;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,x1,x2,x3;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = bs*i;
    s1  = b[idt];  s2 = b[1+idt];s3 = b[2+idt];

    for (k=0; k<nz; k++) {
      idx = bs*vi[k];
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;

      v +=  bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[3]*s2 + v[6]*s3;
    x[1+idt] = v[1]*s1 + v[4]*s2 + v[7]*s3;
    x[2+idt] = v[2]*s1 + v[5]*s2 + v[8]*s3;

  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Special case where the matrix was ILU(0) factored in the natural
   ordering. This eliminates the need for the column and row permutation.
*/
PetscErrorCode MatSolve_SeqBAIJ_2_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  PetscErrorCode    ierr;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,x1,x2;
  const PetscScalar *b;
  PetscInt          jdx,idt,idx,nz,i;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[0]; x[1] = b[1];
  for (i=1; i<n; i++) {
    v    =  aa      + 4*ai[i];
    vi   =  aj      + ai[i];
    nz   =  diag[i] - ai[i];
    idx +=  2;
    s1   =  b[idx];s2 = b[1+idx];
    while (nz--) {
      jdx = 2*(*vi++);
      x1  = x[jdx];x2 = x[1+jdx];
      s1 -= v[0]*x1 + v[2]*x2;
      s2 -= v[1]*x1 + v[3]*x2;
      v  += 4;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 4*diag[i] + 4;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    idt = 2*i;
    s1  = x[idt];  s2 = x[1+idt];
    while (nz--) {
      idx = 2*(*vi++);
      x1  = x[idx];   x2 = x[1+idx];
      s1 -= v[0]*x1 + v[2]*x2;
      s2 -= v[1]*x1 + v[3]*x2;
      v  += 4;
    }
    v        = aa +  4*diag[i];
    x[idt]   = v[0]*s1 + v[2]*s2;
    x[1+idt] = v[1]*s1 + v[3]*s2;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*4*(a->nz) - 2.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_2_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  = a->mbs,*vi,*ai=a->i,*aj=a->j,*adiag=a->diag;
  PetscInt          i,k,nz,idx,idt,jdx;
  PetscErrorCode    ierr;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,x1,x2;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];
  for (i=1; i<n; i++) {
    v   = aa + 4*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = 2*i;
    s1  = b[idx];s2 = b[1+idx];
    PetscPrefetchBlock(vi+nz,nz,0,PETSC_PREFETCH_HINT_NTA);
    PetscPrefetchBlock(v+4*nz,4*nz,0,PETSC_PREFETCH_HINT_NTA);
    for (k=0; k<nz; k++) {
      jdx = 2*vi[k];
      x1  = x[jdx];x2 = x[1+jdx];
      s1 -= v[0]*x1 + v[2]*x2;
      s2 -= v[1]*x1 + v[3]*x2;
      v  +=  4;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 4*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = 2*i;
    s1  = x[idt];  s2 = x[1+idt];
    PetscPrefetchBlock(vi+nz,nz,0,PETSC_PREFETCH_HINT_NTA);
    PetscPrefetchBlock(v+4*nz,4*nz,0,PETSC_PREFETCH_HINT_NTA);
    for (k=0; k<nz; k++) {
      idx = 2*vi[k];
      x1  = x[idx];   x2 = x[1+idx];
      s1 -= v[0]*x1 + v[2]*x2;
      s2 -= v[1]*x1 + v[3]*x2;
      v  += 4;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[2]*s2;
    x[1+idt] = v[1]*s1 + v[3]*s2;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*4*(a->nz) - 2.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqBAIJ_2_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  = a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,k,nz,idx,jdx;
  PetscErrorCode    ierr;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,x1,x2;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];
  for (i=1; i<n; i++) {
    v   = aa + 4*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = 2*i;
    s1  = b[idx];s2 = b[1+idx];
    PetscPrefetchBlock(vi+nz,nz,0,PETSC_PREFETCH_HINT_NTA);
    PetscPrefetchBlock(v+4*nz,4*nz,0,PETSC_PREFETCH_HINT_NTA);
    for (k=0; k<nz; k++) {
      jdx = 2*vi[k];
      x1  = x[jdx];x2 = x[1+jdx];
      s1 -= v[0]*x1 + v[2]*x2;
      s2 -= v[1]*x1 + v[3]*x2;
      v  +=  4;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
  }


  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(4.0*(a->nz) - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqBAIJ_2_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  = a->mbs,*vi,*aj=a->j,*adiag=a->diag;
  PetscInt          i,k,nz,idx,idt;
  PetscErrorCode    ierr;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,x1,x2;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 4*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = 2*i;
    s1  = b[idt];  s2 = b[1+idt];
    PetscPrefetchBlock(vi+nz,nz,0,PETSC_PREFETCH_HINT_NTA);
    PetscPrefetchBlock(v+4*nz,4*nz,0,PETSC_PREFETCH_HINT_NTA);
    for (k=0; k<nz; k++) {
      idx = 2*vi[k];
      x1  = x[idx];   x2 = x[1+idx];
      s1 -= v[0]*x1 + v[2]*x2;
      s2 -= v[1]*x1 + v[3]*x2;
      v  += 4;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[2]*s2;
    x[1+idt] = v[1]*s1 + v[3]*s2;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(4.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Special case where the matrix was ILU(0) factored in the natural
   ordering. This eliminates the need for the column and row permutation.
*/
PetscErrorCode MatSolve_SeqBAIJ_1_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  = a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  PetscErrorCode    ierr;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,x1;
  PetscInt          jdx,idt,idx,nz,i;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[0];
  for (i=1; i<n; i++) {
    v    =  aa      + ai[i];
    vi   =  aj      + ai[i];
    nz   =  diag[i] - ai[i];
    idx +=  1;
    s1   =  b[idx];
    while (nz--) {
      jdx = *vi++;
      x1  = x[jdx];
      s1 -= v[0]*x1;
      v  += 1;
    }
    x[idx] = s1;
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + diag[i] + 1;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    idt = i;
    s1  = x[idt];
    while (nz--) {
      idx = *vi++;
      x1  = x[idx];
      s1 -= v[0]*x1;
      v  += 1;
    }
    v      = aa +  diag[i];
    x[idt] = v[0]*s1;
  }
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*(a->nz) - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatForwardSolve_SeqBAIJ_1_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    n = a->mbs,*ai = a->i,*aj = a->j,*vi;
  PetscScalar       *x,sum;
  const PetscScalar *b;
  const MatScalar   *aa = a->a,*v;
  PetscInt          i,nz;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* forward solve the lower triangular */
  x[0] = b[0];
  v    = aa;
  vi   = aj;
  for (i=1; i<n; i++) {
    nz  = ai[i+1] - ai[i];
    sum = b[i];
    PetscSparseDenseMinusDot(sum,x,v,vi,nz);
    v   += nz;
    vi  += nz;
    x[i] = sum;
  }
  ierr = PetscLogFlops(a->nz - A->cmap->n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqBAIJ_1_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    n = a->mbs,*aj = a->j,*adiag = a->diag,*vi;
  PetscScalar       *x,sum;
  const PetscScalar *b;
  const MatScalar   *aa = a->a,*v;
  PetscInt          i,nz;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + adiag[i+1] + 1;
    vi  = aj + adiag[i+1] + 1;
    nz  = adiag[i] - adiag[i+1]-1;
    sum = b[i];
    PetscSparseDenseMinusDot(sum,x,v,vi,nz);
    x[i] = sum*v[nz]; /* x[i]=aa[adiag[i]]*sum; v++; */
  }

  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_1_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    n = a->mbs,*ai = a->i,*aj = a->j,*adiag = a->diag,*vi;
  PetscScalar       *x,sum;
  const PetscScalar *b;
  const MatScalar   *aa = a->a,*v;
  PetscInt          i,nz;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* forward solve the lower triangular */
  x[0] = b[0];
  v    = aa;
  vi   = aj;
  for (i=1; i<n; i++) {
    nz  = ai[i+1] - ai[i];
    sum = b[i];
    PetscSparseDenseMinusDot(sum,x,v,vi,nz);
    v   += nz;
    vi  += nz;
    x[i] = sum;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + adiag[i+1] + 1;
    vi  = aj + adiag[i+1] + 1;
    nz  = adiag[i] - adiag[i+1]-1;
    sum = x[i];
    PetscSparseDenseMinusDot(sum,x,v,vi,nz);
    x[i] = sum*v[nz]; /* x[i]=aa[adiag[i]]*sum; v++; */
  }

  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

