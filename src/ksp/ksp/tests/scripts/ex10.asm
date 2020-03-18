#! /bin/csh

foreach matrix (arco1)

  foreach np (1 2 3 4)

      foreach blocks (4 5 8 12) 

        foreach overlap (0 1 2 5) 
        echo "matrix $matrix np $np  blocks $blocks overlap $overlap"
          mpiexec -n $np ex10 -f0 /home/bsmith/petsc/src/mat/examples/matrices/$matrix -pc_type asm -mat_mpiaij -pc_asm_blocks $blocks -pc_asm_overlap $overlap

      end

    end

  end

end
