/*     =================================
    Test arrays BGL alignment on 16bit boundary memory

    Date: 06/23/05
    Authors:  Pascal Vezolle/Yannick Langlois 
      ================================= */

int align1(int x) {
   if ( (x & 0xf) == 0 ) return 0;
   return -1;
}
int align2(int x, int x1) {
   if ( ((x | x1) & 0xf) == 0 ) return 0;
   return -1;
}
int align3(int x, int x1, int x2) {
   if ( (((x | x1) | x2) & 0xf) == 0 ) return 0;
   return -1;
}
int align4(int x, int x1, int x2, int x3) {
   if ( ((((x | x1) | x2) | x3) & 0xf) == 0 ) return 0;
   return -1;
}
int align5(int x, int x1, int x2, int x3, int x4) {
   if ( (((((x | x1) | x2) | x3) | x4) & 0xf) == 0 ) return 0;
   return -1;
}
int align6(int x, int x1, int x2, int x3, int x4, int x5) {
   if ( ((((((x | x1) | x2) | x3) | x4) | x5) & 0xf) == 0 ) return 0;
   return -1;
}
int align7(int x, int x1, int x2, int x3, int x4, int x5, int x6) {
   if ( (((((((x | x1) | x2) | x3) | x4) | x5) | x6) & 0xf) == 0 ) return 0;
   return -1;
}
