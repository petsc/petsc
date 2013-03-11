import config.base

import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.sizes = {}
    self.c99_complex = 0
    self.cxx_complex = 0
    return

  def setupHelp(self, help):
    import nargs
    help.addArgument('Types', '-known-endian=<big or little>', nargs.Arg(None, None, 'Are bytes stored in big or little endian?'))
    help.addArgument('Visibility', '-with-visibility=<bool>', nargs.Arg(None, 0, 'Use compiler visibility flags to limit symbol visibility'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    self.headers   = framework.require('config.headers', self)
    return

  def check(self, typeName, defaultType = None):
    '''Checks that "typeName" exists, and if not defines it to "defaultType" if given'''
    self.framework.log.write('Checking for type: '+typeName+'\n')
    include = '''
#include <sys/types.h>
#if STDC_HEADERS
#include <stdlib.h>
#include <stddef.h>
#endif
    '''
    found = self.checkCompile(include,typeName+' a;')
    if not found and defaultType:
      self.addTypedef(defaultType, typeName)
    else:
      self.framework.log.write(typeName+' found\n')
    return found

  def check__int64(self):
    '''Checks if __int64 exists. This is primarily for windows.'''
    if self.check('__int64'):
      self.addDefine('HAVE___INT64',1)
    return

  def checkSizeTypes(self):
    '''Checks for types associated with sizes, such as size_t.'''
    self.check('size_t', 'int')
    return

  def checkIntegerTypes(self):
    '''Checks for types associated with integers, such as int32_t.'''
    self.check('int32_t', 'int')
    return

  def checkFileTypes(self):
    '''Checks for types associated with files, such as mode_t, off_t, etc.'''
    self.check('mode_t', 'int')
    self.check('off_t', 'int')
    return

  def checkPID(self):
    '''Checks for pid_t, and defines it if necessary'''
    return self.check('pid_t', 'int')

  def checkUID(self):
    '''Checks for uid_t and gid_t, and defines them if necessary'''
    if not self.outputPreprocess('sys/types.h').find('uid_t'):
      self.addDefine('uid_t', 'int')
      self.addDefine('gid_t', 'int')
    return

  def checkSignal(self):
    '''Checks the return type of signal() and defines RETSIGTYPE to that type name'''
    includes = '''
#include <sys/types.h>
#include <signal.h>
#ifdef signal
#undef signal
#endif
#ifdef __cplusplus
extern "C" void (*signal (int, void(*)(int)))(int);
#else
void (*signal())();
#endif
    '''
    if self.checkCompile(includes, ''):
      returnType = 'void'
    else:
      returnType = 'int'
    self.addDefine('RETSIGTYPE', returnType)
    return

  def checkC99Complex(self):
    '''Check for complex numbers in in C99 std'''
    includes = '#include <complex.h>\n'
    body     = 'double complex x;\n x = I;\n'
    if self.checkLink(includes, body):
      self.addDefine('HAVE_C99_COMPLEX', 1)
      self.c99_complex = 1
    return

  def checkCxxComplex(self):
    '''Check for complex numbers in namespace std'''
    self.pushLanguage('C++')
    includes = '#include <complex>\n'
    body     = 'std::complex<double> x;\n'
    if self.checkLink(includes, body):
      self.addDefine('HAVE_CXX_COMPLEX', 1)
      self.cxx_complex = 1
    self.popLanguage()
    return

  def checkFortranStar(self):
    '''Checks whether integer*4, etc. is handled in Fortran, and if not defines MISSING_FORTRANSTAR'''
    self.pushLanguage('FC')
    body = '        integer*4 i\n        real*8 d\n'
    if not self.checkCompile('', body):
      self.addDefine('MISSING_FORTRANSTAR', 1)
    self.popLanguage()
    return

# reverse of the above - but more standard thing to do for F90 compilers
  def checkFortranKind(self):
    '''Checks whether selected_int_kind etc work USE_FORTRANKIND'''
    self.pushLanguage('FC')
    body = '''
        integer(kind=selected_int_kind(10)) i
        real(kind=selected_real_kind(10)) d
'''
    if self.checkCompile('', body):
      self.addDefine('USE_FORTRANKIND', 1)
    self.popLanguage()
    return

  def checkFortranDReal(self):
    '''Checks whether dreal is provided in Fortran, and if not defines MISSING_DREAL'''
    self.pushLanguage('FC')
    if not self.checkLink('', '      double precision d\n      d = dreal(3.0)'):
      self.addDefine('MISSING_DREAL', 1)
    self.popLanguage()
    return

  def checkConst(self):
    '''Checks for working const, and if not found defines it to empty string'''
    body = '''
    /* Ultrix mips cc rejects this.  */
    typedef int charset[2]; const charset x;
    /* SunOS 4.1.1 cc rejects this.  */
    char const *const *ccp;
    char **p;
    /* NEC SVR4.0.2 mips cc rejects this.  */
    struct point {int x, y;};
    static struct point const zero = {0,0};
    /* AIX XL C 1.02.0.0 rejects this.
    It does not let you subtract one const X* pointer from another in an arm
    of an if-expression whose if-part is not a constant expression */
    const char *g = "string";
    ccp = &g + (g ? g-g : 0);
    /* HPUX 7.0 cc rejects these. */
    ++ccp;
    p = (char**) ccp;
    ccp = (char const *const *) p;
    /* This section avoids unused variable warnings */
    if (zero.x);
    if (x[0]);
    { /* SCO 3.2v4 cc rejects this.  */
      char *t;
      char const *s = 0 ? (char *) 0 : (char const *) 0;

      *t++ = 0;
      if (*s);
    }
    { /* Someone thinks the Sun supposedly-ANSI compiler will reject this.  */
      int x[] = {25, 17};
      const int *foo = &x[0];
      ++foo;
    }
    { /* Sun SC1.0 ANSI compiler rejects this -- but not the above. */
      typedef const int *iptr;
      iptr p = 0;
      ++p;
    }
    { /* AIX XL C 1.02.0.0 rejects this saying
      "k.c", line 2.27: 1506-025 (S) Operand must be a modifiable lvalue. */
      struct s { int j; const int *ap[3]; };
      struct s *b; b->j = 5;
    }
    { /* ULTRIX-32 V3.1 (Rev 9) vcc rejects this */
      const int foo = 10;

      /* Get rid of unused variable warning */
      if (foo);
    }
    '''
    if not self.checkCompile('', body):
      self.addDefine('const', '')
    return

  def checkEndian(self):
    '''If the machine is big endian, defines WORDS_BIGENDIAN'''
    if 'known-endian' in self.framework.argDB:
      endian = self.framework.argDB['known-endian']
    else:
      # See if sys/param.h defines the BYTE_ORDER macro
      includes = '#include <sys/types.h>\n#ifdef HAVE_SYS_PARAM_H\n  #include <sys/param.h>\n#endif\n'
      body     = '''
#if !BYTE_ORDER || !BIG_ENDIAN || !LITTLE_ENDIAN
  bogus endian macros
#endif
      '''
      if self.checkCompile(includes, body):
        # It does, so check whether it is defined to BIG_ENDIAN or not
        body = '''
#if BYTE_ORDER != BIG_ENDIAN
  not big endian
#endif
        '''
        if self.checkCompile(includes, body):
          endian = 'big'
        else:
          endian = 'little'
      else:
        if not self.framework.argDB['with-batch']:
          body = '''
          /* Are we little or big endian?  From Harbison&Steele. */
          union
          {
            long l;
            char c[sizeof(long)];
          } u;
          u.l = 1;
          exit(u.c[sizeof(long) - 1] == 1);
          '''
          self.pushLanguage('C')
          if self.checkRun('#include <stdlib.h>\n', body, defaultArg = 'isLittleEndian'):
            endian = 'little'
          else:
            endian = 'big'
          self.popLanguage()
        else:
          self.framework.addBatchBody(['{',
                                       '  union {long l; char c[sizeof(long)];} u;',
                                       '  u.l = 1;',
                                       '  fprintf(output, " \'--known-endian=%s\',\\n", (u.c[sizeof(long) - 1] == 1) ? "big" : "little");',
                                       '}'])
          # Dummy value
          endian = 'little'
    if endian == 'big':
      self.addDefine('WORDS_BIGENDIAN', 1)
    return

  def checkSizeof(self, typeName, otherInclude = None):
    '''Determines the size of type "typeName", and defines SIZEOF_"typeName" to be the size'''
    self.framework.log.write('Checking for size of type: '+typeName+'\n')
    filename = 'conftestval'
    includes = '''
#include <sys/types.h>
#if STDC_HEADERS
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#endif\n'''
    mpiFix = '''
#define MPICH_IGNORE_CXX_SEEK
#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1\n'''
    if otherInclude:
      if otherInclude == 'mpi.h':
        includes += mpiFix
      includes += '#include <'+otherInclude+'>\n'
    body     = 'FILE *f = fopen("'+filename+'", "w");\n\nif (!f) exit(1);\nfprintf(f, "%lu\\n", (unsigned long)sizeof('+typeName+'));\n'
    typename = typeName.replace(' ', '-').replace('*', 'p')
    if not 'known-sizeof-'+typename in self.framework.argDB:
      if not self.framework.argDB['with-batch']:
        self.pushLanguage('C')
        if self.checkRun(includes, body) and os.path.exists(filename):
          f    = file(filename)
          size = int(f.read())
          f.close()
          os.remove(filename)
        elif not typename == 'long-long':
          msg = 'Cannot run executable to determine size of '+typeName+'. If this machine uses a batch system \nto submit jobs you will need to configure using ./configure with the additional option  --with-batch.\n Otherwise there is problem with the compilers. Can you compile and run code with your C/C++ (and maybe Fortran) compilers?\n'
          raise RuntimeError(msg)
        else:
          self.framework.log.write('Compiler does not support long long\n')
          size = 0
        self.popLanguage()
      else:
        self.framework.addBatchInclude(['#include <stdlib.h>', '#include <stdio.h>', '#include <sys/types.h>'])
        if otherInclude:
          if otherInclude == 'mpi.h':
            self.framework.addBatchInclude(mpiFix)
          self.framework.addBatchInclude('#include <'+otherInclude+'>')
        self.framework.addBatchBody('fprintf(output, "  \'--known-sizeof-'+typename+'=%d\',\\n", sizeof('+typeName+'));')
        # dummy value
        size = 4
    else:
      size = self.framework.argDB['known-sizeof-'+typename]
    self.sizes['known-sizeof-'+typename] = int(size)
    self.addDefine('SIZEOF_'+typeName.replace(' ', '_').replace('*', 'p').upper(), size)
    return size

  def checkBitsPerByte(self):
    '''Determine the nubmer of bits per byte and define BITS_PER_BYTE'''
    filename = 'conftestval'
    includes = '''
#if STDC_HEADERS
#include <stdlib.h>
#include <stdio.h>
#endif\n'''
    body     = 'FILE *f = fopen("'+filename+'", "w");\n'+'''
    char val[2];
    int i = 0;

    if (!f) exit(1);
    val[0]=\'\\1\';
    val[1]=\'\\0\';
    while(val[0]) {val[0] <<= 1; i++;}
    fprintf(f, "%d\\n", i);\n
    '''
    if 'known-bits-per-byte' in self.framework.argDB:
      bits = self.framework.argDB['known-bits-per-byte']
    elif not self.framework.argDB['with-batch']:
      if self.checkRun(includes, body) and os.path.exists(filename):
        f    = file(filename)
        bits = int(f.read())
        f.close()
        os.remove(filename)
      else:
         msg = 'Cannot run executable to determine bits per bit. If this machine uses a batch system \nto submit jobs you will need to configure using ./configure with the additional option  --with-batch.\n Otherwise there is problem with the compilers. Can you compile and run code with your C/C++ (and maybe Fortran) compilers?\n'
         raise RuntimeError(msg)
    else:
      self.framework.addBatchBody(['{',
                                   '  int i = 0;',
                                   '  char val[2];',
                                   '  val[0]=\'\\1\';',
                                   '  val[1]=\'\\0\';',
                                   '  while(val[0]) {val[0] <<= 1; i++;}',
                                   '  fprintf(output, "  \'--known-bits-per-byte=%d\',\\n", i);',
                                   '}'])
      # dummy value
      bits = 8

    self.bits_per_byte = int(bits)
    self.addDefine('BITS_PER_BYTE', bits)
    return


  def checkVisibility(self):
    if self.framework.argDB['with-visibility']:
      if not self.checkCompile('','__attribute__((visibility ("default"))) int foo(void);'):
        raise RuntimeError('Cannot use visibility attributes')
      self.addDefine('USE_VISIBILITY',1)

  def configure(self):
    self.executeTest(self.check__int64)
    self.executeTest(self.checkSizeTypes)
    self.executeTest(self.checkFileTypes)
    self.executeTest(self.checkIntegerTypes)
    self.executeTest(self.checkPID)
    self.executeTest(self.checkUID)
    self.executeTest(self.checkSignal)
    self.executeTest(self.checkC99Complex)
    if hasattr(self.compilers, 'CXX'):
      self.executeTest(self.checkCxxComplex)
    if hasattr(self.compilers, 'FC'):
      #self.executeTest(self.checkFortranStar)
      self.executeTest(self.checkFortranKind)
      self.executeTest(self.checkFortranDReal)
    self.executeTest(self.checkConst)
    self.executeTest(self.checkEndian)
    map(lambda type: self.executeTest(self.checkSizeof, type), ['char','void *', 'short', 'int', 'long', 'long long', 'float', 'double', 'size_t'])
    self.executeTest(self.checkBitsPerByte)
    self.executeTest(self.checkVisibility)
    return
