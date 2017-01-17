!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        subroutine calculate_B(j1, j2, j, symbol, cutoff, &
                cutofffn, fac_length, factorial, n_symbols, rs, &
                rs_length, psis, psis_length, thetas_length, thetas, &
                 phis, phis_length)
            double precision :: j1, j2, cutoff, j
            integer::  n_symbol, fac_length, phis_length, thetas_length
            integer::  psis_length, rs_length, n_symbols
            character(len=20):: cutofffn, symbol
            double precision, dimension(fac_length)::  factorial
            double precision, dimension(phis_length)::  phis
            double precision, dimension(thetas_length)::  thetas
            double precision, dimension(psis_length)::  psis
            double precision, dimension(rs_length)::  rs
!f2py     intent(in):: j1, j2, j, symbol, cutoff
!f2py     intent(in):: cutofffn, factorial, n_symbols, rs, psis
!f2py     intent(in):: thetas, phis, phis_length, psis_length
!f2py     intent(out):: value
            print*, 'Fortran'
            print*, j1, j2, j, symbol, cutoff, cutofffn, factorial
            print*, n_symbols, rs, psis, thetas, phis
            cutofffn = cutofffn
        end subroutine calculate_B
