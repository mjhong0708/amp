      module cutoffs
          implicit none

          contains
      function cutoff_fxn(r, rc, cutofffn, p_gamma)
          double precision:: r, rc, pi, cutoff_fxn
          ! gamma parameter for the polynomial cutoff
          double precision, optional:: p_gamma
          character(len=20):: cutofffn

!       To avoid noise, for each call of this function, it is better to
!       set returned variables to 0.0d0.
        cutoff_fxn = 0.0d0

        if (cutofffn == 'Cosine') then
            if (r > rc) then
                cutoff_fxn = 0.0d0
            else
                pi = 4.0d0 * datan(1.0d0)
                cutoff_fxn = 0.5d0 * (cos(pi*r/rc) + 1.0d0)
            end if
        elseif (cutofffn == 'Polynomial') then
            if (r > rc) then
                cutoff_fxn = 0.0d0
            else
                cutoff_fxn = 1. + p_gamma &
                      * (r / rc) ** (p_gamma + 1) &
                      - (p_gamma + 1) * (r / rc) ** p_gamma
            end if
        endif
      end function cutoff_fxn

      function cutoff_fxn_prime(r, rc, cutofffn, p_gamma)
          double precision:: r, rc, cutoff_fxn_prime, pi
          ! gamma parameter for the polynomial cutoff
          double precision, optional:: p_gamma
          character(len=20):: cutofffn

!       To avoid noise, for each call of this function, it is better to
!       set returned variables to 0.0d0.
          cutoff_fxn_prime = 0.0d0

          if (cutofffn == 'Cosine') then
              if (r > rc) then
                  cutoff_fxn_prime = 0.0d0
              else
                  pi = 4.0d0 * datan(1.0d0)
                  cutoff_fxn_prime = -0.5d0 * pi * sin(pi*r/rc) / rc
              end if
          elseif (cutofffn == 'Polynomial') then
              if (r > rc) then
                  cutoff_fxn_prime = 0.0d0
              else
                  cutoff_fxn_prime = (p_gamma * (p_gamma + 1) / rc) &
                   * ((r / rc) ** p_gamma - (r / rc) ** (p_gamma - 1))
              end if
          end if
      end function cutoff_fxn_prime

      end module cutoffs
