!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!     Fortran Version = 9
      subroutine check_version(version, warning)
      implicit none

      integer::  version, warning
!f2py         intent(in)::  version
!f2py         intent(out)::  warning
      if (version .NE. 9) then
          warning = 1
      else
          warning = 0
      end if
      end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     module containing all the data of fingerprints (should be fed in
!     by python)
      module fingerprint_props
      implicit none

      integer, allocatable:: num_fingerprints_of_elements(:)
      double precision, allocatable:: raveled_fingerprints(:, :)
      double precision, allocatable:: raveled_fingerprintprimes(:, :)

      end module fingerprint_props

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     module containing model data (should be fed in by python)
      module model_props
      implicit none
      ! mode_signal is 1 for image-centered mode, and 2 for
      ! atom-centered mode
      integer:: mode_signal
      logical:: train_forces
      double precision:: energy_coefficient
      double precision:: force_coefficient
      double precision:: overfit
      logical:: numericprime
      double precision:: d

      end module model_props

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     module containing all the data of images (should be fed in by
!     python)
      module images_props
      implicit none

      integer:: num_images
!     atom-centered variables
      integer:: num_elements
      integer, allocatable:: elements_numbers(:)
      integer, allocatable:: num_images_atoms(:)
      integer, allocatable:: atomic_numbers(:)
      integer, allocatable:: num_neighbors(:)
      integer, allocatable:: raveled_neighborlists(:)
      double precision, allocatable:: actual_energies(:)
      double precision, allocatable:: actual_forces(:, :)
!     image-centered variables
      integer:: num_atoms
      double precision, allocatable:: atomic_positions(:, :)

      end module images_props

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     subroutine that calculates the loss function and its prime
      subroutine calculate_loss(parameters, num_parameters, &
      lossprime, loss, dloss_dparameters, energyloss, forceloss, &
      energy_maxresid, force_maxresid)

      use images_props
      use fingerprint_props
      use model_props
      use neuralnetwork

!!!!!!!!!!!!!!!!!!!!!!!! input/output variables !!!!!!!!!!!!!!!!!!!!!!!!

      integer:: num_parameters
      double precision:: parameters(num_parameters)
      logical:: lossprime
      double precision:: loss, energyloss, forceloss
      double precision:: energy_maxresid, force_maxresid
      double precision:: dloss_dparameters(num_parameters)
!f2py         intent(in):: parameters, num_parameters
!f2py         intent(in):: lossprime
!f2py         intent(out):: loss, energyloss, forceloss
!f2py         intent(out):: energy_maxresid, force_maxresid
!f2py         intent(out):: dloss_dparameters

!!!!!!!!!!!!!!!!!!!!!!!!!!! type definition !!!!!!!!!!!!!!!!!!!!!!!!!!!!

      type:: image_forces
        sequence
        double precision, allocatable:: atom_forces(:, :)
      end type image_forces

      type:: integer_one_d_array
        sequence
        integer, allocatable:: onedarray(:)
      end type integer_one_d_array

      type:: embedded_real_one_one_d_array
        sequence
        type(real_one_d_array), allocatable:: onedarray(:)
      end type embedded_real_one_one_d_array

      type:: embedded_real_one_two_d_array
        sequence
        type(real_two_d_array), allocatable:: onedarray(:)
      end type embedded_real_one_two_d_array

      type:: embedded_integer_one_one_d_array
        sequence
        type(integer_one_d_array), allocatable:: onedarray(:)
      end type embedded_integer_one_one_d_array

      type:: embedded_one_one_two_d_array
        sequence
        type(embedded_real_one_two_d_array), allocatable:: onedarray(:)
      end type embedded_one_one_two_d_array

!!!!!!!!!!!!!!!!!!!!!!!!!! dummy variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      double precision, allocatable::  fingerprint(:)
      type(embedded_real_one_one_d_array), allocatable:: &
      unraveled_fingerprints(:)
      type(integer_one_d_array), allocatable:: &
      unraveled_atomic_numbers(:)
      double precision:: amp_energy, actual_energy, atom_energy
      double precision:: residual_per_atom, dforce, force_resid
      double precision:: overfitloss
      integer:: i, index, j, p, k, q, l, m, &
      len_of_fingerprint, symbol, element, image_no, num_inputs
      double precision:: denergy_dparameters(num_parameters)
      double precision:: daenergy_dparameters(num_parameters)
      double precision:: dforce_dparameters(num_parameters)
      double precision:: doverfitloss_dparameters(num_parameters)
      type(real_two_d_array), allocatable:: dforces_dparameters(:)
      type(image_forces), allocatable:: unraveled_actual_forces(:)
      type(embedded_integer_one_one_d_array), allocatable:: &
      unraveled_neighborlists(:)
      type(embedded_one_one_two_d_array), allocatable:: &
      unraveled_fingerprintprimes(:)
      double precision, allocatable:: fingerprintprime(:)
      integer:: nindex, nsymbol, selfindex
      double precision, allocatable:: &
      actual_forces_(:, :), amp_forces(:, :)
      integer, allocatable:: neighborindices(:)
!     image-centered mode
      type(real_one_d_array), allocatable:: &
      unraveled_atomic_positions(:)
      double precision, allocatable::  inputs(:), inputs_(:)

!!!!!!!!!!!!!!!!!!!!!!!!!!!! calculations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      if (mode_signal == 1) then
        allocate(inputs(3 * num_atoms))
        allocate(inputs_(3 * num_atoms))
        allocate(unraveled_atomic_positions(num_images))
        call unravel_atomic_positions()
      else
        allocate(unraveled_fingerprints(num_images))
        allocate(unraveled_atomic_numbers(num_images))
        allocate(unraveled_neighborlists(num_images))
        allocate(unraveled_fingerprintprimes(num_images))
        call unravel_atomic_numbers()
        call unravel_fingerprints()
      end if
      if (train_forces .EQV. .TRUE.) then
          allocate(unraveled_actual_forces(num_images))
          call unravel_actual_forces()
          if (mode_signal == 2) then
              call unravel_neighborlists()
              call unravel_fingerprintprimes()
          end if
      end if

      energyloss = 0.0d0
      forceloss = 0.0d0
      energy_maxresid = 0.0d0
      force_maxresid = 0.0d0
      do j = 1, num_parameters
        dloss_dparameters(j) = 0.0d0
      end do

!     summation over images
      do image_no = 1, num_images

        if (mode_signal == 1) then
            num_inputs = 3 * num_atoms
            inputs = unraveled_atomic_positions(image_no)%onedarray
        else
            num_atoms = num_images_atoms(image_no)
        end if
        actual_energy = actual_energies(image_no)
        ! calculates amp_energy
        call calculate_energy(image_no)
        ! calculates energy_maxresid
        residual_per_atom = ABS(amp_energy - actual_energy) / num_atoms
        if (residual_per_atom .GT. energy_maxresid) then
            energy_maxresid = residual_per_atom
        end if
        ! calculates energyloss
        energyloss = energyloss + residual_per_atom ** 2.0d0
        if (lossprime .EQV. .TRUE.) then
            ! calculates denergy_dparameters
            if (mode_signal == 1) then ! image-centered mode
                denergy_dparameters = &
                calculate_denergy_dparameters_(num_inputs, inputs, &
                num_parameters, parameters)
            else  ! atom-centered mode
				do j = 1, num_parameters
					denergy_dparameters(j) = 0.0d0
				end do
                if (numericprime .EQV. .FALSE.) then
                    call calculate_denergy_dparameters(image_no)
                else
                    call calculate_numerical_denergy_dparameters(image_no)
                end if
            end if
			! calculates contribution of energyloss to dloss_dparameters
            do j = 1, num_parameters
                dloss_dparameters(j) = dloss_dparameters(j) + &
                energy_coefficient *  2.0d0 * &
                (amp_energy - actual_energy) * &
                denergy_dparameters(j) / (num_atoms ** 2.0d0)
            end do
        end if
        if (train_forces .EQV. .TRUE.) then
            allocate(actual_forces_(num_atoms, 3))
            do selfindex = 1, num_atoms
                do i = 1, 3
                    actual_forces_(selfindex, i) = &
                    unraveled_actual_forces(&
                    image_no)%atom_forces(selfindex, i)
                end do
            end do
            ! calculates amp_forces
            call calculate_forces(image_no)
			! calculates forceloss
            do selfindex = 1, num_atoms
                do i = 1, 3
                    forceloss = forceloss + &
                    (1.0d0 / 3.0d0) * (amp_forces(selfindex, i) - &
                    actual_forces_(selfindex, i)) ** 2.0d0 / num_atoms
                end do
            end do

            ! calculates force_maxresid
            do selfindex = 1, num_atoms
                do i = 1, 3
                    force_resid = &
                    ABS(amp_forces(selfindex, i) - &
                    actual_forces_(selfindex, i))
                    if (force_resid .GT. force_maxresid) then
                        force_maxresid = force_resid
                    end if
                end do
            end do
            if (lossprime .EQV. .TRUE.) then
                allocate(dforces_dparameters(num_atoms))
				do selfindex = 1, num_atoms
					allocate(dforces_dparameters(&
					selfindex)%twodarray(3, num_parameters))
					do i = 1, 3
					    do j = 1, num_parameters
					        dforces_dparameters(&
					        selfindex)%twodarray(i, j) = 0.0d0
					    end do
					end do
				end do
                ! calculates dforces_dparameters
                if (numericprime .EQV. .FALSE.) then
                    call calculate_dforces_dparameters(image_no)
                else
                    call calculate_numerical_dforces_dparameters(image_no)
                end if
                ! calculates contribution of forceloss to
                ! dloss_dparameters
                do selfindex = 1, num_atoms
                    do i = 1, 3
                        do j = 1, num_parameters
                            dloss_dparameters(j) = &
                            dloss_dparameters(j) + &
                            force_coefficient * (2.0d0 / 3.0d0) * &
                            (amp_forces(selfindex, i) - &
                            actual_forces_(selfindex, i)) * &
                            dforces_dparameters(&
                            selfindex)%twodarray(i, j) / num_atoms
                        end do
                    end do
                end do
				do p = 1, size(dforces_dparameters)
					deallocate(dforces_dparameters(p)%twodarray)
				end do
				deallocate(dforces_dparameters)
            end if
            deallocate(actual_forces_)
            deallocate(amp_forces)
        end if
      end do
      loss = energy_coefficient * energyloss + &
             force_coefficient * forceloss

      ! if overfit coefficient is more than zero, overfit
      ! contribution to loss and dloss_dparameters is also added.
      if (overfit .GT. 0.0d0) then
          overfitloss = 0.0d0
          do j = 1, num_parameters
              overfitloss = overfitloss + &
              parameters(j) ** 2.0d0
          end do
          overfitloss = overfit * overfitloss
          loss = loss + overfitloss
          do j = 1, num_parameters
		      doverfitloss_dparameters(j) = &
              2.0d0 * overfit * parameters(j)
              dloss_dparameters(j) = dloss_dparameters(j) + &
              doverfitloss_dparameters(j)
          end do
      end if

!     deallocations for all images
      if (mode_signal == 1) then
        do image_no = 1, num_images
            deallocate(unraveled_atomic_positions(image_no)%onedarray)
        end do
        deallocate(unraveled_atomic_positions)
        deallocate(inputs)
        deallocate(inputs_)
      else
        do image_no = 1, num_images
            deallocate(unraveled_atomic_numbers(image_no)%onedarray)
        end do
        deallocate(unraveled_atomic_numbers)
        do image_no = 1, num_images
            num_atoms = num_images_atoms(image_no)
            do index = 1, num_atoms
                deallocate(unraveled_fingerprints(&
                image_no)%onedarray(index)%onedarray)
            end do
            deallocate(unraveled_fingerprints(image_no)%onedarray)
        end do
        deallocate(unraveled_fingerprints)
      end if

      if (train_forces .EQV. .TRUE.) then
        do image_no = 1, num_images
            deallocate(unraveled_actual_forces(image_no)%atom_forces)
        end do
        deallocate(unraveled_actual_forces)
        if (mode_signal == 2) then
            do image_no = 1, num_images
                num_atoms = num_images_atoms(image_no)
                do selfindex = 1, num_atoms
                    do nindex = 1, &
                    size(unraveled_fingerprintprimes(&
                    image_no)%onedarray(selfindex)%onedarray)
                        deallocate(&
                        unraveled_fingerprintprimes(&
                        image_no)%onedarray(selfindex)%onedarray(&
                        nindex)%twodarray)
                    end do
                    deallocate(unraveled_fingerprintprimes(&
                    image_no)%onedarray(selfindex)%onedarray)
                end do
                deallocate(unraveled_fingerprintprimes(&
                image_no)%onedarray)
            end do
            deallocate(unraveled_fingerprintprimes)
            do image_no = 1, num_images
                num_atoms = num_images_atoms(image_no)
                do index = 1, num_atoms
                    deallocate(unraveled_neighborlists(&
                    image_no)%onedarray(index)%onedarray)
                end do
                deallocate(unraveled_neighborlists(image_no)%onedarray)
            end do
            deallocate(unraveled_neighborlists)
        end if
      end if


      contains

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! calculates amp_energy
      subroutine calculate_energy(image_no)

      if (mode_signal == 1) then
        amp_energy = &
        calculate_image_energy(num_inputs, inputs, num_parameters, &
        parameters)
      else
        amp_energy = 0.0d0
        do index = 1, num_atoms
            symbol = unraveled_atomic_numbers(&
            image_no)%onedarray(index)
            do element = 1, num_elements
                if (symbol == elements_numbers(element)) then
                    exit
                end if
            end do
            len_of_fingerprint = num_fingerprints_of_elements(element)
            allocate(fingerprint(len_of_fingerprint))
            do p = 1, len_of_fingerprint
                fingerprint(p) = &
                unraveled_fingerprints(&
                image_no)%onedarray(index)%onedarray(p)
            end do

            atom_energy = calculate_atomic_energy(symbol, &
            len_of_fingerprint, fingerprint, num_elements, &
            elements_numbers, num_parameters, parameters)
            deallocate(fingerprint)
            amp_energy = amp_energy + atom_energy
        end do
      end if

      end subroutine calculate_energy

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! calculates amp_forces
      subroutine calculate_forces(image_no)

      allocate(amp_forces(num_atoms, 3))
      do selfindex = 1, num_atoms
        do i = 1, 3
            amp_forces(selfindex, i) = 0.0d0
        end do
      end do

      do selfindex = 1, num_atoms
        if (mode_signal == 1) then
            do i = 1, 3
                do p = 1,  3 * num_atoms
                    inputs_(p) = 0.0d0
                end do
                inputs_(3 * (selfindex - 1) + i) = 1.0d0
                amp_forces(selfindex, i) = calculate_force_(num_inputs, &
                inputs, inputs_, num_parameters, parameters)
            end do
        else
            ! neighborindices list is generated.
            allocate(neighborindices(size(&
            unraveled_neighborlists(image_no)%onedarray(&
            selfindex)%onedarray)))
            do p = 1, size(unraveled_neighborlists(&
            image_no)%onedarray(selfindex)%onedarray)
                neighborindices(p) = unraveled_neighborlists(&
                image_no)%onedarray(selfindex)%onedarray(p)
            end do

            do l = 1, size(neighborindices)
                nindex = neighborindices(l)
                nsymbol = unraveled_atomic_numbers(&
                          image_no)%onedarray(nindex)
                do element = 1, num_elements
                    if (nsymbol == elements_numbers(element)) then
                        exit
                    end if
                end do
                len_of_fingerprint = &
                num_fingerprints_of_elements(element)
                allocate(fingerprint(len_of_fingerprint))
                do p = 1, len_of_fingerprint
                    fingerprint(p) = unraveled_fingerprints(&
                    image_no)%onedarray(nindex)%onedarray(p)
                end do

                do i = 1, 3
                    allocate(fingerprintprime(len_of_fingerprint))
                    do p = 1, len_of_fingerprint
                        fingerprintprime(p) = &
                        unraveled_fingerprintprimes(&
                        image_no)%onedarray(&
                        selfindex)%onedarray(l)%twodarray(i, p)
                    end do
                    dforce = calculate_force(nsymbol, len_of_fingerprint, &
                            fingerprint, fingerprintprime, &
                            num_elements, elements_numbers, &
                            num_parameters, parameters)
                    amp_forces(selfindex, i) = &
                    amp_forces(selfindex, i) + dforce
                    deallocate(fingerprintprime)
                end do
                deallocate(fingerprint)
            end do
            deallocate(neighborindices)
        end if
      end do

      end subroutine calculate_forces

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! calculates analytical denergy_dparameters in
      ! the atom-centered mode.
      subroutine calculate_denergy_dparameters(image_no)

      do index = 1, num_atoms
          symbol = unraveled_atomic_numbers(image_no)%onedarray(index)
          do element = 1, num_elements
              if (symbol == elements_numbers(element)) then
                  exit
              end if
          end do
          len_of_fingerprint = num_fingerprints_of_elements(element)
          allocate(fingerprint(len_of_fingerprint))
          do p = 1, len_of_fingerprint
              fingerprint(p) = unraveled_fingerprints(&
              image_no)%onedarray(index)%onedarray(p)
          end do
          daenergy_dparameters = calculate_datomicenergy_dparameters(&
          symbol, len_of_fingerprint, fingerprint, &
          num_elements, elements_numbers, num_parameters, parameters)
          deallocate(fingerprint)
          do j = 1, num_parameters
             denergy_dparameters(j) = denergy_dparameters(j) + &
             daenergy_dparameters(j)
          end do
      end do

      end subroutine calculate_denergy_dparameters

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! calculates numerical denergy_dparameters in the
      ! atom-centered mode.
      subroutine calculate_numerical_denergy_dparameters(image_no)

      double precision:: eplus, eminus

      do j = 1, num_parameters
          parameters(j) = parameters(j) + d
          call calculate_energy(image_no)
          eplus = amp_energy
          parameters(j) = parameters(j) - 2.0d0 * d
          call calculate_energy(image_no)
          eminus = amp_energy
          denergy_dparameters(j) = (eplus - eminus) / (2.0d0 * d)
          parameters(j) = parameters(j) + d
      end do
      call calculate_energy(image_no)

      end subroutine calculate_numerical_denergy_dparameters

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! calculates dforces_dparameters.
      subroutine calculate_dforces_dparameters(image_no)

      if (mode_signal == 1) then ! image-centered mode
          do selfindex = 1, num_atoms
              do i = 1, 3
                do p = 1,  3 * num_atoms
                    inputs_(p) = 0.0d0
                end do
                inputs_(3 * (selfindex - 1) + i) = 1.0d0
                dforce_dparameters = calculate_dforce_dparameters_(&
                num_inputs, inputs, inputs_, num_parameters, parameters)
                do j = 1, num_parameters
                    dforces_dparameters(selfindex)%twodarray(i, j) = &
                    dforce_dparameters(j)
                end do
              end do
          end do

      else ! atom-centered mode
          do selfindex = 1, num_atoms
              ! neighborindices list is generated.
              allocate(neighborindices(size(&
              unraveled_neighborlists(image_no)%onedarray(&
              selfindex)%onedarray)))
              do p = 1, size(unraveled_neighborlists(&
              image_no)%onedarray(selfindex)%onedarray)
                  neighborindices(p) = unraveled_neighborlists(&
                  image_no)%onedarray(selfindex)%onedarray(p)
              end do
              do l = 1, size(neighborindices)
                  nindex = neighborindices(l)
                  nsymbol = unraveled_atomic_numbers(&
                  image_no)%onedarray(nindex)
                  do element = 1, num_elements
                      if (nsymbol == elements_numbers(element)) then
                        exit
                      end if
                  end do
                  len_of_fingerprint = &
                  num_fingerprints_of_elements(element)
                  allocate(fingerprint(len_of_fingerprint))
                  do p = 1, len_of_fingerprint
                      fingerprint(p) = unraveled_fingerprints(&
                      image_no)%onedarray(nindex)%onedarray(p)
                  end do
                  do i = 1, 3
                      allocate(fingerprintprime(len_of_fingerprint))
                      do p = 1, len_of_fingerprint
                          fingerprintprime(p) = &
                          unraveled_fingerprintprimes(&
                          image_no)%onedarray(selfindex)%onedarray(&
                          l)%twodarray(i, p)
                      end do
                      dforce_dparameters = calculate_dforce_dparameters(&
                      nsymbol, len_of_fingerprint, fingerprint, &
                      fingerprintprime, num_elements, &
                      elements_numbers, num_parameters, parameters)
                      deallocate(fingerprintprime)
                      do j = 1, num_parameters
                          dforces_dparameters(&
                          selfindex)%twodarray(i, j) = &
                          dforces_dparameters(&
                          selfindex)%twodarray(i, j) + &
                          dforce_dparameters(j)
                      end do
                  end do
                  deallocate(fingerprint)
              end do
              deallocate(neighborindices)
          end do
      end if

      end subroutine calculate_dforces_dparameters

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ! calculates numerical dforces_dparameters in the
      ! atom-centered mode.
      subroutine calculate_numerical_dforces_dparameters(image_no)

      double precision, allocatable:: fplus(:, :), fminus(:, :)

      do j = 1, num_parameters
          parameters(j) = parameters(j) + d
          deallocate(amp_forces)
          call calculate_forces(image_no)
		  allocate(fplus(num_atoms, 3))
		  do selfindex = 1, num_atoms
			do i = 1, 3
				fplus(selfindex, i) = amp_forces(selfindex, i)
			end do
		  end do
          parameters(j) = parameters(j) - 2.0d0 * d
          deallocate(amp_forces)
          call calculate_forces(image_no)
		  allocate(fminus(num_atoms, 3))
		  do selfindex = 1, num_atoms
			do i = 1, 3
				fminus(selfindex, i) = amp_forces(selfindex, i)
			end do
		  end do
          do selfindex = 1, num_atoms
              do i = 1, 3
                  dforces_dparameters(selfindex)%twodarray(i, j) = &
                  (fplus(selfindex, i) - fminus(selfindex, i)) / &
                  (2.0d0 * d)
              end do
          end do
          parameters(j) = parameters(j) + d
          deallocate(fplus)
          deallocate(fminus)
      end do
      deallocate(amp_forces)
      call calculate_forces(image_no)

      end subroutine calculate_numerical_dforces_dparameters

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!     used only in the image-centered mode.
      subroutine unravel_atomic_positions()

      do image_no = 1, num_images
        allocate(unraveled_atomic_positions(image_no)%onedarray(&
        3 * num_atoms))
        do index = 1, num_atoms
            do i = 1, 3
                unraveled_atomic_positions(image_no)%onedarray(&
                3 * (index - 1) + i) = atomic_positions(&
                image_no, 3 * (index - 1) + i)
             end do
        end do
      end do

      end subroutine unravel_atomic_positions

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_atomic_numbers()

      k = 0
      do image_no = 1, num_images
        num_atoms = num_images_atoms(image_no)
        allocate(unraveled_atomic_numbers(&
        image_no)%onedarray(num_atoms))
        do l = 1, num_atoms
            unraveled_atomic_numbers(image_no)%onedarray(l) &
            = atomic_numbers(k + l)
        end do
        k = k + num_atoms
      end do

      end subroutine unravel_atomic_numbers

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_neighborlists()

      k = 0
      q = 0
      do image_no = 1, num_images
        num_atoms = num_images_atoms(image_no)
        allocate(unraveled_neighborlists(image_no)%onedarray(&
        num_atoms))
        do index = 1, num_atoms
            allocate(unraveled_neighborlists(image_no)%onedarray(&
            index)%onedarray(num_neighbors(k + index)))
            do p = 1, num_neighbors(k + index)
                unraveled_neighborlists(image_no)%onedarray(&
                index)%onedarray(p) = raveled_neighborlists(q + p)+1
            end do
            q = q + num_neighbors(k + index)
        end do
        k = k + num_atoms
      end do

      end subroutine unravel_neighborlists

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_actual_forces()

      k = 0
      do image_no = 1, num_images
        if (mode_signal == 1) then
            num_atoms = num_atoms
        else
            num_atoms = num_images_atoms(image_no)
        end if
        allocate(unraveled_actual_forces(image_no)%atom_forces(&
        num_atoms, 3))
        do index = 1, num_atoms
            do i = 1, 3
                unraveled_actual_forces(image_no)%atom_forces(&
                index, i) = actual_forces(k + index, i)
            end do
        end do
        k = k + num_atoms
      end do

      end subroutine unravel_actual_forces

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_fingerprints()

      k = 0
      do image_no = 1, num_images
        num_atoms = &
        num_images_atoms(image_no)
        allocate(unraveled_fingerprints(&
        image_no)%onedarray(num_atoms))
        do index = 1, num_atoms
            do element = 1, num_elements
                if (unraveled_atomic_numbers(&
                image_no)%onedarray(index)== &
                elements_numbers(element)) then
                    allocate(unraveled_fingerprints(&
                    image_no)%onedarray(index)%onedarray(&
                    num_fingerprints_of_elements(element)))
                    exit
                end if
            end do
            do l = 1, num_fingerprints_of_elements(element)
                unraveled_fingerprints(&
                image_no)%onedarray(index)%onedarray(l) = &
                raveled_fingerprints(k + index, l)
            end do
        end do
      k = k + num_atoms
      end do

      end subroutine unravel_fingerprints

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine unravel_fingerprintprimes()

      integer:: no_of_neighbors

      k = 0
      m = 0
      do image_no = 1, num_images
        num_atoms = &
        num_images_atoms(image_no)
        allocate(unraveled_fingerprintprimes(&
        image_no)%onedarray(num_atoms))
        do selfindex = 1, num_atoms
			! neighborindices list is generated.
            allocate(neighborindices(size(unraveled_neighborlists(&
            image_no)%onedarray(selfindex)%onedarray)))
            do p = 1, size(unraveled_neighborlists(&
            image_no)%onedarray(selfindex)%onedarray)
                neighborindices(p) = unraveled_neighborlists(&
                image_no)%onedarray(selfindex)%onedarray(p)
            end do
            no_of_neighbors = num_neighbors(k + selfindex)
            allocate(unraveled_fingerprintprimes(&
            image_no)%onedarray(selfindex)%onedarray(no_of_neighbors))
            do nindex = 1, no_of_neighbors
                do nsymbol = 1, num_elements
                if (unraveled_atomic_numbers(&
                image_no)%onedarray(neighborindices(nindex)) == &
                elements_numbers(nsymbol)) then
                    exit
                end if
                end do
                allocate(unraveled_fingerprintprimes(&
                image_no)%onedarray(selfindex)%onedarray(&
                nindex)%twodarray(3, num_fingerprints_of_elements(&
                nsymbol)))
                do p = 1, 3
                    do q = 1, num_fingerprints_of_elements(nsymbol)
                        unraveled_fingerprintprimes(&
                        image_no)%onedarray(selfindex)%onedarray(&
                        nindex)%twodarray(p, q) = &
                        raveled_fingerprintprimes(&
                        3 * m + 3 * nindex + p - 3, q)
                    end do
                end do
            end do
            deallocate(neighborindices)
            m = m + no_of_neighbors
        end do
        k = k + num_atoms
      end do

      end subroutine unravel_fingerprintprimes

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      end subroutine calculate_loss

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!     subroutine that deallocates variables
      subroutine deallocate_variables()

      use images_props
      use fingerprint_props
      use model_props
      use neuralnetwork

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!     deallocating fingerprint_props
      if (allocated(num_fingerprints_of_elements) .EQV. .TRUE.) then
        deallocate(num_fingerprints_of_elements)
      end if
      if (allocated(raveled_fingerprints) .EQV. .TRUE.) then
        deallocate(raveled_fingerprints)
      end if
      if (allocated(raveled_fingerprintprimes) .EQV. .TRUE.) then
        deallocate(raveled_fingerprintprimes)
      end if

!     deallocating images_props
      if (allocated(elements_numbers) .EQV. .TRUE.) then
        deallocate(elements_numbers)
      end if
      if (allocated(num_images_atoms) .EQV. .TRUE.) then
        deallocate(num_images_atoms)
      end if
      if (allocated(atomic_numbers) .EQV. .TRUE.) then
        deallocate(atomic_numbers)
      end if
      if (allocated(num_neighbors) .EQV. .TRUE.) then
        deallocate(num_neighbors)
      end if
      if (allocated(raveled_neighborlists) .EQV. .TRUE.) then
        deallocate(raveled_neighborlists)
      end if
      if (allocated(actual_energies) .EQV. .TRUE.) then
        deallocate(actual_energies)
      end if
      if (allocated(actual_forces) .EQV. .TRUE.) then
        deallocate(actual_forces)
      end if
      if (allocated(atomic_positions) .EQV. .TRUE.) then
        deallocate(atomic_positions)
      end if

!     deallocating neuralnetwork
      if (allocated(min_fingerprints) .EQV. .TRUE.) then
        deallocate(min_fingerprints)
      end if
      if (allocated(max_fingerprints) .EQV. .TRUE.) then
        deallocate(max_fingerprints)
      end if
      if (allocated(no_layers_of_elements) .EQV. .TRUE.) then
        deallocate(no_layers_of_elements)
      end if
      if (allocated(no_nodes_of_elements) .EQV. .TRUE.) then
        deallocate(no_nodes_of_elements)
      end if

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      end subroutine deallocate_variables

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
