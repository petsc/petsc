#if !defined(__TAO_VERSION_H)
#define __TAO_VERSION_H

/* ========================================================================== */
/* 
   Current TAO version number and release date, also listed in
    docs/changes.html
    docs/tex/manual/manual.tex 
    docs/tex/manual/intro.tex 
    docs/tex/manual/manual_tex.tex
*/
#define TAO_VERSION_NUMBER "TAO Version 2.0-beta"
#define TAO_VERSION_RELEASE  0 // 1 if official release
#define TAO_VERSION_MAJOR    2
#define TAO_VERSION_MINOR    0
#define TAO_VERSION_SUBMINOR 0
#define TAO_PATCH_LEVEL      0
#define TAO_VERSION_(MAJOR,MINOR,SUBMINOR) \
    ((TAO_VERSION_MAJOR == (MAJOR)) &&       \
    (TAO_VERSION_MINOR == (MINOR)) &&       \
     (TAO_VERSION_SUBMINOR == (SUBMINOR)))
#define TAO_VERSION_DATE     "Jun 30, 2011"
#define TAO_AUTHOR_INFO      "The TAO Team:\
 Todd Munson, Jorge More', Jason Sarich\n\
Bug reports, questions: tao-comments@mcs.anl.gov\n\
Web page: http://www.mcs.anl.gov/tao/\n"

#endif
