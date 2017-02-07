
/*T
      description: test of subtests and test parsing generates lots of small tests
T*/
 
int main(int argc, char **argv) {return 0;}

/*TEST
   test: 
     # Test separate_testvars at the top lvel
     suffix: 1
     command: echo ${args} @SUBARGS@
     nsize: {{1,2}}
     args: -a b -c {{d "e,f" g}} -h {{"i,j" "i,j"}} -l {{foo bar}}
     separate_testvars: l

   test: # Simple subtest
     suffix: 2
     command: echo ${args} @SUBARGS@

     args: -a b -c {{d "e,f" g}} 

     test:
       args:  -my_arg cg -my_arg2 {{p q}}

     test:
       args: -foo {{"s,t" "u,v"}} -rtol {{1.e-10 1.e-11}}

   test: # subtests with suffix to generate separate files
     suffix: 3
     command: echo ${args} @SUBARGS@

     args: -a b -c {{d "e,f" g}}

     test:
       suffix: cg
       args:  -my_arg cg

     test:
       suffix: gmres
       args:  -my_arg gmres -rtol {{1.e-10 1.e-11}}

   test: # Everything separate test generation added
     suffix: 4
     command: echo ${args} @SUBARGS@

     args: -a b -c {{d "e,f" g}}

     test:
       suffix: cg
       args:  -my_arg cg 

     test:
       suffix: gmres
       args: -rtol {{1.e-10 1.e-11}}
       separate_testvars: rtol
TEST*/
