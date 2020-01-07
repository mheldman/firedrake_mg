for solver in gmg pfas
  do
  for problem in ice radial
    do
    for probtype in obstacle mse plaplace ice
    do
      for numlevels in 5 6 7 8 9 10
      do
        for smooth in 1 2
        do
          for cycle in V FV
          do
              echo $solver $problem $probtype $numlevels $smooth $cycle
              python test.py -d zz$problem -o ./testresults/$problem:$numlevels:$probtype:$solver:$cycle:$smooth --numlevels $numlevels --mgtype $solver --cycle $cycle --maxiters 300 --rtol 1e-10 --probtype $probtype --preiters $smooth --postiters $smooth
              if [$numlevels -eq 10]
              then
                python test.py -d $problem -o ./testresults/$problem:$numlevels:$probtype:$solver:$cycle:$smooth:fmg --numlevels 10 --mgtype $solver --cycle $cycle --maxiters 300 --rtol 1e-10 --fmgc 2 --probtype $probtype --preiters $smooth --postiters $smooth
              fi
            done
          done
        done
      done
    done
  done
done

