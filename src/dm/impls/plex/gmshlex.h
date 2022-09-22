/*
 S: simplex  B: box
 N: size     I: index  L: loop
 p: degree (aka order in Gmsh)
 1,2,3: topological dimension
 i,j,k: coordinate indices
*/

#define SN1(p)          ((p) + 1)
#define SN2(p)          (SN1(p) * SN1((p) + 1) / 2)
#define SN3(p)          (SN2(p) * SN1((p) + 2) / 3)
#define SI1(p, i)       ((i))
#define SI2(p, i, j)    ((i) + (SN2(p) - SN2((p) - (j))))
#define SI3(p, i, j, k) (SI2((p) - (k), i, j) + (SN3(p) - SN3((p) - (k))))
#define SL1(p, i)       for ((i) = 1; (i) < (p); ++(i))
#define SL2(p, i, j)    SL1((p)-1, i) SL1((p) - (i), j)
#define SL3(p, i, j, k) SL1((p)-2, i) SL1((p) - (i), j) SL1((p) - (i) - (j), k)

#define BN1(p)          ((p) + 1)
#define BN2(p)          (BN1(p) * BN1(p))
#define BN3(p)          (BN2(p) * BN1(p))
#define BI1(p, i)       ((i))
#define BI2(p, i, j)    ((i) + (j)*BN1(p))
#define BI3(p, i, j, k) ((i) + BI2(p, j, k) * BN1(p))
#define BL1(p, i)       for ((i) = 1; (i) < (p); ++(i))
#define BL2(p, i, j)    BL1(p, i) BL1(p, j)
#define BL3(p, i, j, k) BL1(p, i) BL1(p, j) BL1(p, k)

#define GmshNumNodes_VTX(p) (1)
#define GmshNumNodes_SEG(p) SN1(p)
#define GmshNumNodes_TRI(p) SN2(p)
#define GmshNumNodes_QUA(p) BN2(p)
#define GmshNumNodes_TET(p) SN3(p)
#define GmshNumNodes_HEX(p) BN3(p)
#define GmshNumNodes_PRI(p) (SN2(p) * BN1(p))
#define GmshNumNodes_PYR(p) (((p) + 1) * ((p) + 2) * (2 * (p) + 3) / 6)

#define GMSH_MAX_ORDER 10

static inline int GmshLexOrder_VTX(int p, int lex[], int node)
{
  lex[0] = node++;
  (void)p;
  return node;
}

static inline int GmshLexOrder_SEG(int p, int lex[], int node)
{
#define loop1(i) SL1(p, i)
#define index(i) SI1(p, i)
  int i;
  /* trivial case */
  if (p == 0) lex[0] = node++;
  if (p == 0) return node;
  /* vertex nodes */
  lex[index(0)] = node++;
  lex[index(p)] = node++;
  if (p == 1) return node;
  /* internal cell nodes */
  loop1(i) lex[index(i)] = node++;
  return node;
#undef loop1
#undef index
}

static inline int GmshLexOrder_TRI(int p, int lex[], int node)
{
#define loop1(i)    SL1(p, i)
#define loop2(i, j) SL2(p, i, j)
#define index(i, j) SI2(p, i, j)
  int i, j, *sub, buf[SN2(GMSH_MAX_ORDER)];
  /* trivial case */
  if (p == 0) lex[0] = node++;
  if (p == 0) return node;
  /* vertex nodes */
  lex[index(0, 0)] = node++;
  lex[index(p, 0)] = node++;
  lex[index(0, p)] = node++;
  if (p == 1) return node;
  /* internal edge nodes */
  loop1(i) lex[index(i, 0)]     = node++;
  loop1(j) lex[index(p - j, j)] = node++;
  loop1(j) lex[index(0, p - j)] = node++;
  if (p == 2) return node;
  /* internal cell nodes */
  node                         = GmshLexOrder_TRI(p - 3, sub = buf, node);
  loop2(j, i) lex[index(i, j)] = *sub++;
  return node;
#undef loop1
#undef loop2
#undef index
}

static inline int GmshLexOrder_QUA(int p, int lex[], int node)
{
#define loop1(i)    BL1(p, i)
#define loop2(i, j) BL2(p, i, j)
#define index(i, j) BI2(p, i, j)
  int i, j, *sub, buf[BN2(GMSH_MAX_ORDER)];
  /* trivial case */
  if (p == 0) lex[0] = node++;
  if (p == 0) return node;
  /* vertex nodes */
  lex[index(0, 0)] = node++;
  lex[index(p, 0)] = node++;
  lex[index(p, p)] = node++;
  lex[index(0, p)] = node++;
  if (p == 1) return node;
  /* internal edge nodes */
  loop1(i) lex[index(i, 0)]     = node++;
  loop1(j) lex[index(p, j)]     = node++;
  loop1(i) lex[index(p - i, p)] = node++;
  loop1(j) lex[index(0, p - j)] = node++;
  /* internal cell nodes */
  node                         = GmshLexOrder_QUA(p - 2, sub = buf, node);
  loop2(j, i) lex[index(i, j)] = *sub++;
  return node;
#undef loop1
#undef loop2
#undef index
}

static inline int GmshLexOrder_TET(int p, int lex[], int node)
{
#define loop1(i)       SL1(p, i)
#define loop2(i, j)    SL2(p, i, j)
#define loop3(i, j, k) SL3(p, i, j, k)
#define index(i, j, k) SI3(p, i, j, k)
  int i, j, k, *sub, buf[SN3(GMSH_MAX_ORDER)];
  /* trivial case */
  if (p == 0) lex[0] = node++;
  if (p == 0) return node;
  /* vertex nodes */
  lex[index(0, 0, 0)] = node++;
  lex[index(p, 0, 0)] = node++;
  lex[index(0, p, 0)] = node++;
  lex[index(0, 0, p)] = node++;
  if (p == 1) return node;
  /* internal edge nodes */
  loop1(i) lex[index(i, 0, 0)]     = node++;
  loop1(j) lex[index(p - j, j, 0)] = node++;
  loop1(j) lex[index(0, p - j, 0)] = node++;
  loop1(k) lex[index(0, 0, p - k)] = node++;
  loop1(j) lex[index(0, j, p - j)] = node++;
  loop1(i) lex[index(i, 0, p - i)] = node++;
  if (p == 2) return node;
  /* internal face nodes */
  node                                    = GmshLexOrder_TRI(p - 3, sub = buf, node);
  loop2(i, j) lex[index(i, j, 0)]         = *sub++;
  node                                    = GmshLexOrder_TRI(p - 3, sub = buf, node);
  loop2(k, i) lex[index(i, 0, k)]         = *sub++;
  node                                    = GmshLexOrder_TRI(p - 3, sub = buf, node);
  loop2(j, k) lex[index(0, j, k)]         = *sub++;
  node                                    = GmshLexOrder_TRI(p - 3, sub = buf, node);
  loop2(j, i) lex[index(i, j, p - i - j)] = *sub++;
  if (p == 3) return node;
  /* internal cell nodes */
  node                               = GmshLexOrder_TET(p - 4, sub = buf, node);
  loop3(k, j, i) lex[index(i, j, k)] = *sub++;
  return node;
#undef loop1
#undef loop2
#undef loop3
#undef index
}

static inline int GmshLexOrder_HEX(int p, int lex[], int node)
{
#define loop1(i)       BL1(p, i)
#define loop2(i, j)    BL2(p, i, j)
#define loop3(i, j, k) BL3(p, i, j, k)
#define index(i, j, k) BI3(p, i, j, k)
  int i, j, k, *sub, buf[BN3(GMSH_MAX_ORDER)];
  /* trivial case */
  if (p == 0) lex[0] = node++;
  if (p == 0) return node;
  /* vertex nodes */
  lex[index(0, 0, 0)] = node++;
  lex[index(p, 0, 0)] = node++;
  lex[index(p, p, 0)] = node++;
  lex[index(0, p, 0)] = node++;
  lex[index(0, 0, p)] = node++;
  lex[index(p, 0, p)] = node++;
  lex[index(p, p, p)] = node++;
  lex[index(0, p, p)] = node++;
  if (p == 1) return node;
  /* internal edge nodes */
  loop1(i) lex[index(i, 0, 0)]     = node++;
  loop1(j) lex[index(0, j, 0)]     = node++;
  loop1(k) lex[index(0, 0, k)]     = node++;
  loop1(j) lex[index(p, j, 0)]     = node++;
  loop1(k) lex[index(p, 0, k)]     = node++;
  loop1(i) lex[index(p - i, p, 0)] = node++;
  loop1(k) lex[index(p, p, k)]     = node++;
  loop1(k) lex[index(0, p, k)]     = node++;
  loop1(i) lex[index(i, 0, p)]     = node++;
  loop1(j) lex[index(0, j, p)]     = node++;
  loop1(j) lex[index(p, j, p)]     = node++;
  loop1(i) lex[index(p - i, p, p)] = node++;
  /* internal face nodes */
  node                                = GmshLexOrder_QUA(p - 2, sub = buf, node);
  loop2(i, j) lex[index(i, j, 0)]     = *sub++;
  node                                = GmshLexOrder_QUA(p - 2, sub = buf, node);
  loop2(k, i) lex[index(i, 0, k)]     = *sub++;
  node                                = GmshLexOrder_QUA(p - 2, sub = buf, node);
  loop2(j, k) lex[index(0, j, k)]     = *sub++;
  node                                = GmshLexOrder_QUA(p - 2, sub = buf, node);
  loop2(k, j) lex[index(p, j, k)]     = *sub++;
  node                                = GmshLexOrder_QUA(p - 2, sub = buf, node);
  loop2(k, i) lex[index(p - i, p, k)] = *sub++;
  node                                = GmshLexOrder_QUA(p - 2, sub = buf, node);
  loop2(j, i) lex[index(i, j, p)]     = *sub++;
  /* internal cell nodes */
  node                               = GmshLexOrder_HEX(p - 2, sub = buf, node);
  loop3(k, j, i) lex[index(i, j, k)] = *sub++;
  return node;
#undef loop1
#undef loop2
#undef loop3
#undef index
}

static inline int GmshLexOrder_PRI(int p, int lex[], int node)
{
#define loop1(i)       BL1(p, i)
#define loops(i, j)    SL2(p, i, j)
#define loopb(i, j)    BL2(p, i, j)
#define index(i, j, k) (SI2(p, i, j) + BI1(p, k) * SN2(p))
  int i, j, k, *sub, buf[BN2(GMSH_MAX_ORDER)];
  /* trivial case */
  if (p == 0) lex[0] = node++;
  if (p == 0) return node;
  /* vertex nodes */
  lex[index(0, 0, 0)] = node++;
  lex[index(p, 0, 0)] = node++;
  lex[index(0, p, 0)] = node++;
  lex[index(0, 0, p)] = node++;
  lex[index(p, 0, p)] = node++;
  lex[index(0, p, p)] = node++;
  if (p == 1) return node;
  /* internal edge nodes */
  loop1(i) lex[index(i, 0, 0)]     = node++;
  loop1(j) lex[index(0, j, 0)]     = node++;
  loop1(k) lex[index(0, 0, k)]     = node++;
  loop1(j) lex[index(p - j, j, 0)] = node++;
  loop1(k) lex[index(p, 0, k)]     = node++;
  loop1(k) lex[index(0, p, k)]     = node++;
  loop1(i) lex[index(i, 0, p)]     = node++;
  loop1(j) lex[index(0, j, p)]     = node++;
  loop1(j) lex[index(p - j, j, p)] = node++;
  if (p >= 3) {
    /* internal bottom face nodes */
    node                            = GmshLexOrder_TRI(p - 3, sub = buf, node);
    loops(i, j) lex[index(i, j, 0)] = *sub++;
    /* internal top face nodes */
    node                            = GmshLexOrder_TRI(p - 3, sub = buf, node);
    loops(j, i) lex[index(i, j, p)] = *sub++;
  }
  if (p >= 2) {
    /* internal front face nodes */
    node                            = GmshLexOrder_QUA(p - 2, sub = buf, node);
    loopb(k, i) lex[index(i, 0, k)] = *sub++;
    /* internal left face nodes */
    node                            = GmshLexOrder_QUA(p - 2, sub = buf, node);
    loopb(j, k) lex[index(0, j, k)] = *sub++;
    /* internal back face nodes */
    node                                = GmshLexOrder_QUA(p - 2, sub = buf, node);
    loopb(k, j) lex[index(p - j, j, k)] = *sub++;
  }
  if (p >= 3) {
    /* internal cell nodes */
    typedef struct {
      int i, j;
    } pair;
    pair ij[SN2(GMSH_MAX_ORDER)], tmp[SN2(GMSH_MAX_ORDER)];
    int  m = GmshLexOrder_TRI(p - 3, sub = buf, 0), l = 0;
    loops(j, i)
    {
      tmp[l].i = i;
      tmp[l].j = j;
      l++;
    }
    for (l = 0; l < m; ++l) ij[sub[l]] = tmp[l];
    for (l = 0; l < m; ++l) {
      i                            = ij[l].i;
      j                            = ij[l].j;
      node                         = GmshLexOrder_SEG(p - 2, sub = buf, node);
      loop1(k) lex[index(i, j, k)] = *sub++;
    }
  }
  return node;
#undef loop1
#undef loops
#undef loopb
#undef index
}

static inline int GmshLexOrder_PYR(int p, int lex[], int node)
{
  int i, m = GmshNumNodes_PYR(p);
  for (i = 0; i < m; ++i) lex[i] = node++; /* TODO */
  return node;
}
