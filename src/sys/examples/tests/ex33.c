
/*T
#  SKIP: test of subtests and test parsing generates lots of small tests
T*/
 
int main(int argc, char **argv) {return 0;}

/*TEST
   test: # this is a meta test of the test parser
     command: echo ${args} ${subargs}
     nsize: {{1,2}}
     args: -a b -c {{d,"e,f",g}} -h {{"i,j","i,j"}} -l {{k,k}}

     test:
       args: -my_arg {{"m,n",o}} -my_arg2 {{p,p}}
       output_file: output/ex33_@C@_@MYVAR@.out

     test:
       args: -foo {{"s,t","s,t"}} -bar {{q,r}}
       output_file: output/ex33_@C@_@MYVAR@.out

TEST*/

