
/*T
  SKIP: test of subtests and test parsing generates lots of small tests
T*/
 
int main(int argc, char **argv) {return 0;}

/*TEST

   test: # this is a meta test of the test parser
     command: echo ${args} ${subargs}
     a: b
     c: {{d,"e,f",g}}
     nsize: {{1,2}}
     args: -a @A@ -c @C@ -h {{"i,j","i,j"}} -l {{k,k}}
     test:
       myvar: {{"m,n",o}}
       args: -my_arg @MYVAR@ -my_arg2 {{p,p}}
       output_file: output/ex33_@C@_@MYVAR@.out
     test:
       myvar: {{q,r}}
       args: -foo {{"s,t","s,t"}} -bar @MYVAR@
       output_file: output/ex33_@C@_@MYVAR@.out
TEST*/

