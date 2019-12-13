#ifndef RELAXATION_H
#define RELAXATION_H

/*
 *  Perform one iteration of Gauss-Seidel relaxation on the linear
 *  system Ax = b, where A is stored in CSR format and x and b
 *  are column vectors.
 *
 *  The unknowns are swept through according to the slice defined
 *  by row_start, row_end, and row_step.  These options are used
 *  to implement standard forward and backward sweeps, or sweeping
 *  only a subset of the unknowns.  A forward sweep is implemented
 *  with gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1) where N is the
 *  number of rows in matrix A.  Similarly, a backward sweep is
 *  implemented with gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1).
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      row_start  - beginning of the sweep
 *      row_stop   - end of the sweep (i.e. one past the last unknown)
 *      row_step   - stride used during the sweep (may be negative)
 *
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */
template<class I, class T, class F>
void gauss_seidel(const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                        T  x[], const int  x_size,
                  const T  b[], const int  b_size,
                  const I row_start,
                  const I row_stop,
                  const I row_step)
{
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        T rsum = 0;
        T diag = 0;

        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }

        if (diag != (F) 0.0){
            x[i] = (b[i] - rsum)/diag;
        }
    }
}


/*
 *  Perform one iteration of projected Gauss-Seidel relaxation on the linear
 *  complementarity system b >= Ax, x >= 0, (b - Ax).T*x = 0, where A is stored in
 *  CSR format and x and b are column vectors.
 *
 *  The unknowns are swept through according to the slice defined
 *  by row_start, row_end, and row_step.  These options are used
 *  to implement standard forward and backward sweeps, or sweeping
 *  only a subset of the unknowns.  A forward sweep is implemented
 *  with gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1) where N is the
 *  number of rows in matrix A.  Similarly, a backward sweep is
 *  implemented with gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1).
 *
 *  Parameters
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      x[]        - approximate solution
 *      b[]        - right hand side
 *      row_start  - beginning of the sweep
 *      row_stop   - end of the sweep (i.e. one past the last unknown)
 *      row_step   - stride used during the sweep (may be negative)
 *
 *  Returns:
 *      Nothing, x will be modified in place
 *
 */

template<class I, class T, class F>
void projected_gauss_seidel(const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const F Ax[], const int Ax_size,
                        F  x[], const int  x_size,
                  const F  b[], const int  b_size,
                  const F  p[], const int  p_size,
                  const I row_start,
                  const I row_stop,
                  const I row_step)
{
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        F rsum = 0;
        F diag = 0;

        for(I jj = start; jj < end; jj++){
            I j = Aj[jj];
            if (i == j)
                diag  = Ax[jj];
            else
                rsum += Ax[jj]*x[j];
        }

        if (diag != (F) 0.0){
            x[i] = (b[i] - rsum)/diag;
            if(x[i] < p[i])
          {
              x[i] = p[i];
          }
        }
    }
}


template<class I, class T, class F>
void truncated_gauss_seidel(const I Ap[], const int Ap_size,
                            const I Aj[], const int Aj_size,
                            const F Ax[], const int Ax_size,
                                  F  x[], const int  x_size,
                            const F  b[], const int  b_size,
                            const I  p[], const int  p_size,
                            const I row_start,
                            const I row_stop,
                            const I row_step)
          {
          
    for(I i = row_start; i != row_stop; i += row_step) {
        if(p[i] == 1){
          I start = Ap[i];
          I end   = Ap[i+1];
          F rsum = 0;
          F diag = 0;

          for(I jj = start; jj < end; jj++){
              I j = Aj[jj];
              if (i == j)
                  diag  = Ax[jj];
              else
                  rsum += Ax[jj]*x[j];
          }

          if (diag != (F) 0.0){
              x[i] = (b[i] - rsum)/diag;
              
          }
        }
    }
}

template<class I, class T, class F>
void icp_gauss_seidel(const I Ap[], const int Ap_size,
                            const I Aj[], const int Aj_size,
                            const F Ax[], const int Ax_size,
                                  F  x[], const int  x_size,
                            const F  b[], const int  b_size,
                                  F  p[], const int  p_size,
                            const I row_start,
                            const I row_stop,
                            const I row_step,
                            const F k)
          {
        
    
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        F rsum = 0;
        F diag = 0;
        if(end - start == 1)
          x[i] = b[i];
        else{
          for(I jj = start; jj < end; jj++){
              I j = Aj[jj];
              if (i == j)
                  diag  = Ax[jj];
              else
                  rsum += Ax[jj]*x[j];
          }
      
      
          if(diag != (F) 0.0)
              x[i] = (b[i] - rsum)/diag;
          if(1 - k*diag != (F) 0.0){
            F val = (k*(rsum - b[i]) + p[i])/(1 - k*diag);
            if(x[i] < val)
              x[i] = val;
            }
          
      
      }
  }
}

template<class I, class T, class F>
void icp_gauss_seidel_jump(const I Ap[], const int Ap_size,
                            const I Aj[], const int Aj_size,
                            const F Ax[], const int Ax_size,
                                  F  x[], const int  x_size,
                            const F  b[], const int  b_size,
                                  F  p[], const int  p_size,
                                  F jumps[], const int jumps_size,
                            const I row_start,
                            const I row_stop,
                            const I row_step,
                            const F k)
          {
        
    
    for(I i = row_start; i != row_stop; i += row_step) {
        I start = Ap[i];
        I end   = Ap[i+1];
        F rsum = 0;
        F diag = 0;
        if(end - start > 1){
          for(I jj = start; jj < end; jj++){
              I j = Aj[jj];
              if (i == j)
                  diag  = Ax[jj];
              else
                  rsum += Ax[jj]*x[j];
          }
      
      
          if(diag != (F) 0.0)
              x[i] = (b[i] - rsum)/diag;
          if(1 - k*diag != (F) 0.0){
            F val = (k*(rsum - b[i]) + p[i])/(1 - k*diag);
            if(x[i] < val)
            {
              jumps[i] = val - x[i];
              x[i] = val;
              }
            }
        else
          x[i] = b[i];
          
      
      }
  }
}


template<class I, class T, class F>
void licp_gauss_seidel( const I Ap[], const int Ap_size,
                        const I Aj[], const int Aj_size,
                        const F Ax[], const int Ax_size,
                        const I Bp[], const int Bp_size,
                        const I Bj[], const int Bj_size,
                        const F Bx[], const int Bx_size,
                              F  x[], const int  x_size,
                        const F  b[], const int  b_size,
                        const F  c[], const int  c_size,
                        const I offsets[], const int offsets_size,
                        const I row_start,
                        const I row_stop,
                        const I row_step)
          {
        
  
    for(I i = row_start; i != row_stop; i += row_step) {
    
      I start = Ap[i];
      I end   = Ap[i+1];
    
      if(end - start == 1)
        x[i] = b[i];
    
      else{
        F rsumA = 0.0;
        F diagA = 0.0;
        for(I jj = start; jj < end; jj++){
          I j = Aj[jj];
          if (i == j)
              diagA  = Ax[jj];
          else
              rsumA += Ax[jj]*x[j];
      
          if(diagA != (F) 0.0)
            x[i] = (b[i] - rsumA)/diagA;
        }
      
  
        for(I k = 0; k < offsets_size; k++){
          I rnum = i + offsets[k];
          if(rnum > -1 && rnum < x_size){
            F diagB = 0.0;
            F rsumB = 0.0;
            start = Bp[rnum];
            end   = Bp[rnum + 1];
            for(I jj = start; jj < end; jj++){
              I j = Bj[jj];
              if (i == j)
                diagB = Bx[jj];
              else
                rsumB += Bx[jj]*x[j];
            }
            if(diagB != (F) 0.0){
              F val = (c[rnum] - rsumB)/diagB;
              if(diagB < (F) 0.0 && val > x[i])
                x[i] = val;
              else if(diagB > (F) 0.0 && val < x[i])
                x[i] = val;
              }
            }
          }
        }
      }
  }

template<class I, class T, class F>
void licpgs_softmax(    const I Ap[], const int Ap_size,
                        const I Aj[], const int Aj_size,
                        const F Ax[], const int Ax_size,
                        const I Bp[], const int Bp_size,
                        const I Bj[], const int Bj_size,
                        const F Bx[], const int Bx_size,
                              F  x[], const int  x_size,
                        const F  b[], const int  b_size,
                        const F  c[], const int  c_size,
                        const I offsets[], const int offsets_size,
                        const I row_start,
                        const I row_stop,
                        const I row_step)
          {
        
  
    for(I i = row_start; i != row_stop; i += row_step) {
    
      I start = Ap[i];
      I end   = Ap[i+1];
    
      if(end - start == 1)
        x[i] = b[i];
    
      else{
        F rsumA = 0.0;
        F diagA = 0.0;
        for(I jj = start; jj < end; jj++){
          I j = Aj[jj];
          if (i == j)
              diagA  = Ax[jj];
          else
              rsumA += Ax[jj]*x[j];
      
          if(diagA != (F) 0.0)
            x[i] = (b[i] - rsumA)/diagA;
        }
  
        for(I k = 0; k < offsets_size; k++){
          I rnum = i + offsets[k];
          if(rnum > -1 && rnum < x_size){
            F diagB = 0.0;
            F rsumB = 0.0;
            start = Bp[rnum];
            end   = Bp[rnum + 1];
            for(I jj = start; jj < end; jj++){
              I j = Bj[jj];
              if (i == j)
                diagB = Bx[jj];
              else
                rsumB += Bx[jj]*x[j];
            }
            
            if(diagB != (F) 0.0){
              F val = (c[rnum] - rsumB)/diagB;
              F maximum;
              F minimum;
              if(x[i] > val){
                maximum = x[i];
                minimum = val;
              }
              else{
                maximum = val;
                minimum = x[i];
              }
              if(x_size != 66049)
	              x[i] = maximum + log(x_size*x_size + exp(x_size*x_size*(minimum - maximum) ))/(x_size*x_size);
              else
                x[i] = maximum;
              }
            }
          }
        }
      }
  }







#endif
