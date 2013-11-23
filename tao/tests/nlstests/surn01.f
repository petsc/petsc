      real function surn01(iseed)
      integer iseed
c     **********
c
c     function surn01
c
c     Rand is the portable random number generator of L. Schrage.
c
c     The generator is full cycle, that is, every integer from
c     1 to 2**31 - 2 is generated exactly once in the cycle.
c     It is completely described in TOMS 5(1979),132-138.
c
c     The function statement is
c
c       real function surn01(iseed)
c
c     where
c
c       iseed is a positive integer variable.
c         On input iseed specifies the seed to the generator.
c         On output the seed is updated.
c
c     MINPACK-2 Project. March 1981.
c     Argonne National Laboratory.
c     Jorge J. More'.
c
c     **********
      integer a, b15, b16, fhi, k, leftlo, p, xhi, xalo
      real c

c     Set a = 7**5, b15 = 2**15, b16 = 2**16, p = 2**31-1, c = 1/p.

      data a/16807/, b15/32768/, b16/65536/, p/2147483647/
      data c/4.656612875e-10/

c     There are 8 steps in surn01.

c     1. Get 15 hi order bits of iseed.
c     2. Get 16 lo bits of iseed and form lo product.
c     3. Get 15 hi order bits of lo product.
c     4. Form the 31 highest bits of full product.
c     5. Get overflo past 31st bit of full product.
c     6. Assemble all the parts and pre-substract p.
c        The parentheses are essential.
c     7. Add p back if necessary.
c     8. Multiply by 1/(2**31-1).

      xhi = iseed/b16
      xalo = (iseed-xhi*b16)*a
      leftlo = xalo/b16
      fhi = xhi*a + leftlo
      k = fhi/b15
      iseed = (((xalo-leftlo*b16)-p)+(fhi-k*b15)*b16) + k
      if (iseed .lt. 0) iseed = iseed + p
      surn01 = c*float(iseed)

      end
