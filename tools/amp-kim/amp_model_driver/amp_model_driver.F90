
module amp_model_driver

use neuralnetwork
use, intrinsic :: iso_c_binding
use kim_model_driver_headers_module
implicit none

private
public model_compute, &
       model_refresh, &
       model_destroy, &
       model_compute_arguments_create, &
       model_compute_arguments_destroy, &
       model_buffer_type, &
       cd

integer(c_int), parameter :: cd = c_double ! used for literal constants

type :: symmetry_function_type
  character(len=8, kind=c_char) :: symmetry_type
  character(len=8, kind=c_char) :: species1_name_string
  type(kim_species_name_type) :: species1_name
  integer(c_int) :: species1_code
  character(len=8, kind=c_char) :: species2_name_string
  type(kim_species_name_type) :: species2_name
  integer(c_int) :: species2_code
  real(c_double) :: eta
  real(c_double) :: gamma
  real(c_double) :: zeta
  real(c_double) :: max_fingerprint
  real(c_double) :: min_fingerprint
end type symmetry_function_type

type :: species_symmetry_function_type
  character(len=8, kind=c_char) :: center_atom_species
  integer(c_int) :: center_atom_species_code
  type(symmetry_function_type), allocatable :: gs(:)
end type species_symmetry_function_type

type :: species_model_type
integer(c_int) :: no_hiddenlayers_of_species
integer(c_int), allocatable :: no_nodes_of_species(:)
character(len=8, kind=c_char) :: activation_function
end type species_model_type

type :: model_buffer_type
  real(c_double) :: influence_distance
  real(c_double) :: cutoff(1)
  integer(c_int) :: padding_neighbor_hints(1)
  integer(c_int) :: half_list_hints(1)

  integer(c_int) :: num_species
  character(len=8, kind=c_char), allocatable :: species_name_strings(:)
  type(kim_species_name_type), allocatable :: species_names(:)
  integer(c_int), allocatable :: species_codes(:)
  integer(c_int), allocatable :: num_symmetry_functions_of_species(:)
  type(species_symmetry_function_type), allocatable :: symmetry_functions(:)
  type(species_model_type), allocatable :: model_props(:)
  real(c_double), allocatable :: parameters(:)

end type model_buffer_type

contains

!-------------------------------------------------------------------------------
#include "kim_model_compute_log_macros.fd"
subroutine model_compute(model_compute_handle, &
  model_compute_arguments_handle, ierr) bind(c)
  implicit none

  type(kim_model_compute_handle_type), intent(in) :: model_compute_handle
  type(kim_model_compute_arguments_handle_type), intent(in) :: &
    model_compute_arguments_handle
  integer(c_int), intent(out) :: ierr

  integer(c_int) :: index, number_of_neighbors, p, q, count, l, num_gs, symbol
  integer(c_int) :: number_of_neighbor_of_neighbors
  integer(c_int) :: selfindex, selfsymbol, nindex, nsymbol, nnindex, nnsymbol
  integer(c_int) :: ierr2
  type(model_buffer_type), pointer :: buf; type(c_ptr) :: pbuf
  integer(c_int), pointer :: num_atoms
  real(c_double), pointer :: energy
  real(c_double), pointer :: coor(:,:)
  real(c_double), pointer:: forces(:, :)
  integer(c_int), pointer :: neighbors_of_particle(:)
  integer(c_int), pointer :: particle_species_codes(:)
  integer(c_int), pointer :: particle_contributing(:)

  integer(c_int), allocatable:: neighbor_numbers(:), neighbor_indices(:)
  real(c_double), allocatable:: neighbor_positions(:,:)
  integer(c_int), allocatable:: neighbor_of_neighbor_numbers(:), neighbor_of_neighbor_indices(:)
  real(c_double), allocatable:: neighbor_of_neighbor_positions(:,:)
  real(c_double), allocatable:: fingerprint(:)
  real(c_double) :: atom_energy
  real(c_double), allocatable:: fingerprintprime(:)
  real(c_double):: dforce
  integer(c_int):: cutofffn_code
  real(c_double):: rc
  real(c_double), dimension(3):: ri

!!!!!!!!!!!!!!!!!!!!!!!!!!! type definition !!!!!!!!!!!!!!!!!!!!!!!!!!!!

  type:: integer_one_d_array
    sequence
    integer(c_int), allocatable:: onedarray(:)
  end type integer_one_d_array

  type:: embedded_real_one_two_d_array
    sequence
    type(real_two_d_array), allocatable:: onedarray(:)
  end type embedded_real_one_two_d_array

  type:: real_one_d_array
    sequence
    real(c_double), allocatable:: onedarray(:)
  end type real_one_d_array

!!!!!!!!!!!!!!!!!!!!!!!!!! dummy variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  type(integer_one_d_array), allocatable :: neighborlists(:)
  type(real_one_d_array), allocatable :: fingerprints(:)
  type(embedded_real_one_two_d_array), allocatable:: fingerprintprimes(:)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  kim_log_file = __FILE__

  ! get model buffer from model_compute object
  call kim_model_compute_get_model_buffer_pointer(model_compute_handle, pbuf)
  call c_f_pointer(pbuf, buf)

  ! Unpack data from model_compute_arguments object
  ierr = 0
  call kim_model_compute_arguments_get_argument_pointer( &
    model_compute_arguments_handle, &
    kim_compute_argument_name_number_of_particles, num_atoms, ierr2)
  ierr = ierr + ierr2
  call kim_model_compute_arguments_get_argument_pointer( &
    model_compute_arguments_handle, &
    kim_compute_argument_name_particle_species_codes, &
    num_atoms, particle_species_codes, ierr2)
  ierr = ierr + ierr2
  call kim_model_compute_arguments_get_argument_pointer( &
    model_compute_arguments_handle, &
    kim_compute_argument_name_particle_contributing, num_atoms, &
    particle_contributing, ierr2)
  ierr = ierr + ierr2
  call kim_model_compute_arguments_get_argument_pointer( &
    model_compute_arguments_handle, &
    kim_compute_argument_name_coordinates, 3, num_atoms, coor, ierr2)
  ierr = ierr + ierr2
  call kim_model_compute_arguments_get_argument_pointer( &
    model_compute_arguments_handle, &
    kim_compute_argument_name_partial_energy, energy, ierr2)
  ierr = ierr + ierr2
  call kim_model_compute_arguments_get_argument_pointer( &
    model_compute_arguments_handle, &
    kim_compute_argument_name_partial_forces, 3, num_atoms, forces, ierr2)
  ierr = ierr + ierr2
  if (ierr /= 0) then
    kim_log_message = "Unable to get all argument pointers"
    LOG_ERROR()
    return
  end if

  ! Check to be sure that the species are correct
  do index = 1, num_atoms
    ierr = 1 ! assumes an error
    do p = 1, buf%num_species
      if (particle_species_codes(index) .eq. buf%species_codes(p)) then
        ierr = 0
        exit
      end if
    end do
    if (ierr .ne. 0) then
      kim_log_message = "Unexpected species code detected"
      LOG_ERROR()
      return
    end if
  end do

  ! Allocate and set up neighborlists for particles
  allocate(neighborlists(num_atoms))
  do index = 1, num_atoms
    call kim_model_compute_arguments_get_neighbor_list( &
    model_compute_arguments_handle, 1, index, number_of_neighbors, &
    neighbors_of_particle, ierr)
    allocate(neighborlists(index)%onedarray(number_of_neighbors))
    do p = 1, number_of_neighbors
      neighborlists(index)%onedarray(p) = neighbors_of_particle(p)
    end do
  end do

  ! Allocate and set up fingerprints of particles
  allocate(fingerprints(num_atoms))
  rc = buf%cutoff(1)
  cutofffn_code = 1 ! for 'Cosine'
  do index = 1, num_atoms
    symbol = particle_species_codes(index)
    num_gs = size(buf%symmetry_functions(symbol)%gs)
    ri = coor(:, index)
    number_of_neighbors = size(neighborlists(index)%onedarray)
    allocate(neighbor_numbers(number_of_neighbors))
    allocate(neighbor_positions(number_of_neighbors, 3))
    do p = 1, number_of_neighbors
      nindex = neighborlists(index)%onedarray(p)
      neighbor_numbers(p) = particle_species_codes(nindex)
      neighbor_positions(p, 1) = coor(1, nindex)
      neighbor_positions(p, 2) = coor(2, nindex)
      neighbor_positions(p, 3) = coor(3, nindex)
    end do
    allocate(fingerprint(num_gs))
    call calculate_fingerprint(num_gs, buf%symmetry_functions(symbol)%gs, &
    number_of_neighbors, neighbor_numbers, &
    neighbor_positions, ri, rc, cutofffn_code, fingerprint)
    allocate(fingerprints(index)%onedarray(num_gs))
    do p = 1, num_gs
      fingerprints(index)%onedarray(p) = fingerprint(p)
    end do
    deallocate(neighbor_numbers)
    deallocate(neighbor_positions)
    deallocate(fingerprint)
  end do

  ! Allocate and set up fingerprintprimes of particles
  allocate(fingerprintprimes(num_atoms))
  do selfindex = 1, num_atoms
    number_of_neighbors = size(neighborlists(selfindex)%onedarray)
    allocate(fingerprintprimes(selfindex)%onedarray(number_of_neighbors))
    do p = 1, number_of_neighbors
      nindex = neighborlists(selfindex)%onedarray(p)
      nsymbol = particle_species_codes(nindex)
      num_gs = size(buf%symmetry_functions(nsymbol)%gs)
      allocate(fingerprintprimes(selfindex)%onedarray(p)%twodarray(3, num_gs))

      number_of_neighbor_of_neighbors = size(neighborlists(nindex)%onedarray)
      allocate(neighbor_of_neighbor_numbers(number_of_neighbor_of_neighbors))
      allocate(neighbor_of_neighbor_indices(number_of_neighbor_of_neighbors))
      allocate(neighbor_of_neighbor_positions(number_of_neighbor_of_neighbors, 3))
      do q = 1, number_of_neighbor_of_neighbors
        nnindex = neighborlists(nindex)%onedarray(q)
        nnsymbol = particle_species_codes(nnindex)
        neighbor_of_neighbor_indices(q) = nnindex
        neighbor_of_neighbor_numbers(q) = nnsymbol
        neighbor_of_neighbor_positions(q, 1) = coor(1, nnindex)
        neighbor_of_neighbor_positions(q, 2) = coor(2, nnindex)
        neighbor_of_neighbor_positions(q, 3) = coor(3, nnindex)
      end do
      do l = 0, 2
        allocate(fingerprintprime(num_gs))
        call calculate_fingerprintprime(num_gs, buf%symmetry_functions(nsymbol)%gs, &
        number_of_neighbor_of_neighbors, neighbor_of_neighbor_indices, neighbor_of_neighbor_numbers, &
        neighbor_of_neighbor_positions, rc, cutofffn_code, nindex, coor(:, nindex), selfindex, l, fingerprintprime)
        do q = 1, num_gs
          fingerprintprimes(selfindex)%onedarray(p)%twodarray(l+1, q) = fingerprintprime(q)
        end do
        deallocate(fingerprintprime)
      end do
      deallocate(neighbor_of_neighbor_numbers)
      deallocate(neighbor_of_neighbor_indices)
      deallocate(neighbor_of_neighbor_positions)
    end do
  end do

  ! As of now, the code only works if the number of fingerprints for different species are the same.
  allocate(min_fingerprints(buf%num_species, size(buf%symmetry_functions(1)%gs)))
  allocate(max_fingerprints(buf%num_species, size(buf%symmetry_functions(1)%gs)))
  do p = 1, buf%num_species
    do q = 1, size(buf%symmetry_functions(p)%gs)
      min_fingerprints(p, q) = buf%symmetry_functions(p)%gs(q)%min_fingerprint
      max_fingerprints(p, q) = buf%symmetry_functions(p)%gs(q)%max_fingerprint
    end do
  end do
  allocate(no_layers_of_elements(buf%num_species))
  do p = 1, buf%num_species
    no_layers_of_elements(p) = buf%model_props(p)%no_hiddenlayers_of_species + 2
  end do
  count = 0
  do p = 1, buf%num_species
    count = count + buf%model_props(p)%no_hiddenlayers_of_species + 2
  end do
  allocate(no_nodes_of_elements(count))
  count = 1
  do p = 1, buf%num_species
    no_nodes_of_elements(count) = size(buf%symmetry_functions(p)%gs)
    count = count + 1
    do q = 1, buf%model_props(p)%no_hiddenlayers_of_species
      no_nodes_of_elements(count) = buf%model_props(p)%no_nodes_of_species(q)
      count = count + 1
    end do
    no_nodes_of_elements(count) = 1
    count = count + 1
  end do
  ! As of now, activation function should be the same for different species neural nets.
  if (buf%model_props(1)%activation_function == 'tanh') then
    activation_signal = 1
  else if (buf%model_props(1)%activation_function == 'sigmoid') then
    activation_signal = 2
  else if (buf%model_props(1)%activation_function == 'linear') then
    activation_signal = 3
  end if

  ! Initialize energy
  energy = 0.0_cd
  !  Loop over particles and compute energy
  do selfindex = 1, num_atoms
    if (particle_contributing(selfindex) == 1) then
      selfsymbol = particle_species_codes(selfindex)
      atom_energy = calculate_atomic_energy(selfsymbol, &
      size(buf%symmetry_functions(selfsymbol)%gs), &
      fingerprints(selfindex)%onedarray, buf%num_species, &
      buf%species_codes, size(buf%parameters), buf%parameters)
      energy = energy + atom_energy
    end if
  end do

  ! Initialize forces
  do selfindex = 1, num_atoms
    do p = 1, 3
      forces(p, selfindex) = 0.0d0
    end do
  end do
  !  Loop over particles and their neighbors and compute forces
  do selfindex = 1, num_atoms
    if (particle_contributing(selfindex) == 1) then

      ! First the contribution of self particle on itself is calculated for forces
      selfsymbol = particle_species_codes(selfindex)
      num_gs = size(buf%symmetry_functions(selfsymbol)%gs)
      number_of_neighbors = size(neighborlists(selfindex)%onedarray)
      allocate(neighbor_numbers(number_of_neighbors))
      allocate(neighbor_indices(number_of_neighbors))
      allocate(neighbor_positions(number_of_neighbors, 3))
      do p = 1, number_of_neighbors
        nindex = neighborlists(selfindex)%onedarray(p)
        neighbor_indices(p) = nindex
        neighbor_numbers(p) = particle_species_codes(nindex)
        neighbor_positions(p, 1) = coor(1, nindex)
        neighbor_positions(p, 2) = coor(2, nindex)
        neighbor_positions(p, 3) = coor(3, nindex)
      end do
      allocate(fingerprint(num_gs))
      do q = 1, num_gs
        fingerprint(q) = fingerprints(selfindex)%onedarray(q)
      end do
      do l = 0, 2
        allocate(fingerprintprime(num_gs))
        call calculate_fingerprintprime(num_gs, buf%symmetry_functions(selfsymbol)%gs, &
        number_of_neighbors, neighbor_indices, neighbor_numbers, &
        neighbor_positions, rc, cutofffn_code, selfindex, coor(:, selfindex), selfindex, l, &
        fingerprintprime)
        dforce = calculate_force(selfsymbol, num_gs, &
        fingerprint, fingerprintprime, &
        buf%num_species, buf%species_codes, &
        size(buf%parameters), buf%parameters)
        forces(l + 1, selfindex) = forces(l + 1, selfindex) + dforce
        deallocate(fingerprintprime)
      end do
      deallocate(fingerprint)
      deallocate(neighbor_numbers)
      deallocate(neighbor_indices)
      deallocate(neighbor_positions)

      ! Second the contribution of neighbors on self particle is calculated for forces
      do p = 1, number_of_neighbors
        nindex = neighborlists(selfindex)%onedarray(p)
        nsymbol = particle_species_codes(nindex)
        num_gs = size(buf%symmetry_functions(nsymbol)%gs)
        allocate(fingerprint(num_gs))
        do q = 1, num_gs
          fingerprint(q) = fingerprints(nindex)%onedarray(q)
        end do
        do l = 0, 2
          allocate(fingerprintprime(num_gs))
          do q = 1, num_gs
            fingerprintprime(q) = &
            fingerprintprimes(selfindex)%onedarray(p)%twodarray(l + 1, q)
          end do
          dforce = calculate_force(nsymbol, num_gs, &
          fingerprint, fingerprintprime, &
          buf%num_species, buf%species_codes, &
          size(buf%parameters), buf%parameters)
          forces(l + 1, selfindex) = forces(l + 1, selfindex) + dforce
          deallocate(fingerprintprime)
        end do
        deallocate(fingerprint)
      end do
    end if
  end do

  deallocate(min_fingerprints)
  deallocate(max_fingerprints)
  deallocate(no_layers_of_elements)
  deallocate(no_nodes_of_elements)

  ! Deallocate fingerprintprimes of particles
  do selfindex = 1, num_atoms
    number_of_neighbors = size(neighborlists(selfindex)%onedarray)
    do p = 1, number_of_neighbors
      deallocate(fingerprintprimes(selfindex)%onedarray(p)%twodarray)
    end do
    deallocate(fingerprintprimes(selfindex)%onedarray)
  end do
  deallocate(fingerprintprimes)

  ! Deallocate fingerprints of particles
  do index = 1, num_atoms
    deallocate(fingerprints(index)%onedarray)
  end do
  deallocate(fingerprints)

  ! Deallocate neighborlist of particles
  do index = 1, num_atoms
    deallocate(neighborlists(index)%onedarray)
  end do
  deallocate(neighborlists)

  ierr = 0  ! Everything is great

end subroutine model_compute


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine calculate_fingerprintprime(num_gs, gs, number_of_neighbors, neighbor_indices, neighbor_numbers, &
neighbor_positions, rc, cutofffn_code, i, ri, m, l, fingerprintprime)

implicit none

integer(c_int):: num_gs, number_of_neighbors, i, m, l, q
type(symmetry_function_type) :: gs(num_gs), g
integer(c_int), dimension(number_of_neighbors):: neighbor_numbers, neighbor_indices
real(c_double), dimension(number_of_neighbors, 3):: neighbor_positions
real(c_double), dimension(3):: ri
real(c_double)::  rc, ridge
integer(c_int):: cutofffn_code
real(c_double):: fingerprintprime(num_gs)
integer(c_int), dimension(2):: g_numbers

do q = 1, num_gs
  g = gs(q)
  if (g%symmetry_type == 'g2') then
    call calculate_g2_prime(neighbor_indices, &
    neighbor_numbers, neighbor_positions, g%species1_code, &
    g%eta, rc, cutofffn_code, i, ri, m, l, number_of_neighbors, ridge)
  else if (g%symmetry_type == 'g4') then
    g_numbers(1) = g%species1_code
    g_numbers(2) = g%species2_code
    call calculate_g4_prime(neighbor_indices, neighbor_numbers, neighbor_positions, &
    g_numbers, g%gamma, g%zeta, g%eta, rc, cutofffn_code, i, ri, m, l, number_of_neighbors, ridge)
  else if (g%symmetry_type == 'g5') then
    g_numbers(1) = g%species1_code
    g_numbers(2) = g%species2_code
    call calculate_g5_prime(neighbor_indices, neighbor_numbers, neighbor_positions, &
    g_numbers, g%gamma, g%zeta, g%eta, rc, cutofffn_code, i, ri, m, l, number_of_neighbors, ridge)
  else
    print *, "Unknown symmetry function type! Only 'g2', 'g4', and 'g5' are supported."
  end if
  fingerprintprime(q) = ridge
end do

end subroutine calculate_fingerprintprime


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine calculate_fingerprint(num_gs, gs, number_of_neighbors, neighbor_numbers, &
neighbor_positions, ri, rc, cutofffn_code, fingerprint)

implicit none

integer(c_int):: num_gs, number_of_neighbors, q
type(symmetry_function_type) :: gs(num_gs), g
integer(c_int), dimension(number_of_neighbors):: neighbor_numbers
real(c_double), dimension(number_of_neighbors, 3):: neighbor_positions
real(c_double), dimension(3):: ri
real(c_double)::  rc, ridge
integer(c_int):: cutofffn_code
real(c_double), dimension(num_gs):: fingerprint
integer(c_int), dimension(2):: g_numbers

do q = 1, num_gs
  g = gs(q)
  if (g%symmetry_type == 'g2') then
    call calculate_g2(neighbor_numbers, neighbor_positions, &
    g%species1_code, g%eta, g%gamma, rc, cutofffn_code, ri, number_of_neighbors, ridge)
  else if (g%symmetry_type == 'g4') then
    g_numbers(1) = g%species1_code
    g_numbers(2) = g%species2_code
    call calculate_g4(neighbor_numbers, neighbor_positions, &
    g_numbers, g%gamma, g%zeta, g%eta, rc, cutofffn_code, ri, number_of_neighbors, ridge)
  else if (g%symmetry_type == 'g5') then
    g_numbers(1) = g%species1_code
    g_numbers(2) = g%species2_code
    call calculate_g5(neighbor_numbers, neighbor_positions, &
    g_numbers, g%gamma, g%zeta, g%eta, rc, cutofffn_code, ri, number_of_neighbors, ridge)
  else
    print *, "Unknown symmetry function type! Only 'g2', 'g4', and 'g5' are supported."
  end if
  fingerprint(q) = ridge
end do

end subroutine calculate_fingerprint

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!-------------------------------------------------------------------------------

! No need to make any changes to the model_destroy() routine
#include "kim_model_destroy_log_macros.fd"
subroutine model_destroy(model_destroy_handle, ierr) bind(c)
  implicit none

  type(kim_model_destroy_handle_type), intent(inout) :: model_destroy_handle
  integer(c_int), intent(out) :: ierr

  type(model_buffer_type), pointer :: buf; type(c_ptr) :: pbuf

  kim_log_file = __FILE__

  call kim_model_destroy_get_model_buffer_pointer(model_destroy_handle, pbuf)
  call c_f_pointer(pbuf, buf)
  kim_log_message = "deallocating model buffer"
  LOG_INFORMATION()
  deallocate(buf)

  ierr = 0  ! everything is good
end subroutine model_destroy

!-------------------------------------------------------------------------------
#include "kim_model_refresh_log_macros.fd"
subroutine model_refresh(model_refresh_handle, ierr) bind(c)
  implicit none

  type(kim_model_refresh_handle_type), intent(inout) :: model_refresh_handle
  integer(c_int), intent(out) :: ierr

  type(model_buffer_type), pointer :: buf; type(c_ptr) :: pbuf

  kim_log_file = __FILE__

  call kim_model_refresh_get_model_buffer_pointer(model_refresh_handle, pbuf)
  call c_f_pointer(pbuf, buf)

  ! Nothing to be done.  This model driver does not publish any parameters

  call kim_model_refresh_set_influence_distance_pointer(model_refresh_handle, &
    buf%influence_distance)
  call kim_model_refresh_set_neighbor_list_pointers(model_refresh_handle, &
    1, buf%cutoff, buf%padding_neighbor_hints)
    !, buf%half_list_hints)

  ierr = 0  ! everything is good
end subroutine model_refresh

!-------------------------------------------------------------------------------
#include "kim_model_compute_arguments_create_log_macros.fd"
subroutine model_compute_arguments_create(model_compute_handle, &
  model_compute_arguments_create_handle, ierr) bind(c)
  use kim_model_compute_arguments_create_module, &
    log_entry => kim_model_compute_arguments_create_log_entry
  implicit none

  type(kim_model_compute_handle_type), intent(in) :: model_compute_handle
  type(kim_model_compute_arguments_create_handle_type), intent(inout) :: &
    model_compute_arguments_create_handle
  integer(c_int), intent(out) :: ierr
  integer(c_int) :: ierr2

  kim_log_file = __FILE__

  ! register arguments
  call kim_model_compute_arguments_create_set_argument_support_status( &
    model_compute_arguments_create_handle, &
    kim_compute_argument_name_partial_energy, &
    kim_support_status_required, ierr)
  call kim_model_compute_arguments_create_set_argument_support_status( &
  model_compute_arguments_create_handle, &
  kim_compute_argument_name_partial_forces, &
  kim_support_status_required, ierr2)
  ierr = ierr + ierr2
  if (ierr /= 0) then
    kim_log_message = "Unable to register arguments support_statuses"
    LOG_ERROR()
    return
  end if

  ierr = 0  ! everything is good
end subroutine model_compute_arguments_create

!-------------------------------------------------------------------------------
#include "kim_model_compute_arguments_destroy_log_macros.fd"
subroutine model_compute_arguments_destroy(model_compute_handle, &
  model_compute_arguments_destroy_handle, ierr) bind(c)
  use kim_model_compute_arguments_destroy_module, &
    log_entry => kim_model_compute_arguments_destroy_log_entry
  implicit none

  type(kim_model_compute_handle_type), intent(in) :: model_compute_handle
  type(kim_model_compute_arguments_destroy_handle_type), intent(inout) :: &
    model_compute_arguments_destroy_handle
  integer(c_int), intent(out) :: ierr

  kim_log_file = __FILE__

  ! nothing to be done

  ierr = 0  ! everything is good
end subroutine model_compute_arguments_destroy

end module amp_model_driver

!-------------------------------------------------------------------------------
#include "kim_model_driver_create_log_macros.fd"
subroutine model_driver_create(model_driver_create_handle, &
  requested_length_unit, requested_energy_unit, requested_charge_unit, &
  requested_temperature_unit, requested_time_unit, ierr) bind(c)
use, intrinsic :: iso_c_binding
use amp_model_driver
use kim_model_driver_headers_module
implicit none

type(kim_model_driver_create_handle_type), intent(inout) :: &
  model_driver_create_handle
type(kim_length_unit_type), intent(in), value :: requested_length_unit
type(kim_energy_unit_type), intent(in), value :: requested_energy_unit
type(kim_charge_unit_type), intent(in), value :: requested_charge_unit
type(kim_temperature_unit_type), intent(in), value :: requested_temperature_unit
type(kim_time_unit_type), intent(in), value :: requested_time_unit
integer(c_int), intent(out) :: ierr

integer(c_int) :: ierr2
integer(c_int) :: number_of_parameter_files
character(len=1024, kind=c_char) :: parameter_file_name
real(c_double) :: unit_conversion_factor
type(model_buffer_type), pointer :: buffer
integer(c_int) :: p, q, k, count
character(len=8, kind=c_char) :: junk

kim_log_file = __FILE__

! use requested units (we'll convert parameters as needed below)
! We only make use of length and energy, other are unused
call kim_model_driver_create_set_units( &
  model_driver_create_handle, &
  requested_length_unit, &
  requested_energy_unit, &
  kim_charge_unit_unused, &
  kim_temperature_unit_unused, &
  kim_time_unit_unused, ierr)
if (ierr /= 0) then
  kim_log_message = "Unable to set units"
  LOG_ERROR()
  return
end if

! we'll use one-based numbering
call kim_model_driver_create_set_model_numbering( &
  model_driver_create_handle, kim_numbering_one_based, ierr)
if (ierr /= 0) then
  kim_log_message = "Unable to set numbering"
  LOG_ERROR()
  return
end if

! store callback pointers in KIM object
call kim_model_driver_create_set_compute_pointer( &
  model_driver_create_handle, kim_language_name_fortran, &
  c_funloc(model_compute), ierr)
call kim_model_driver_create_set_compute_arguments_create_pointer( &
  model_driver_create_handle, kim_language_name_fortran, &
  c_funloc(model_compute_arguments_create), ierr2)
ierr = ierr + ierr2
call kim_model_driver_create_set_compute_arguments_destroy_pointer( &
  model_driver_create_handle, kim_language_name_fortran, &
  c_funloc(model_compute_arguments_destroy), ierr2)
ierr = ierr + ierr2
call kim_model_driver_create_set_refresh_pointer( &
  model_driver_create_handle, kim_language_name_fortran, &
  c_funloc(model_refresh), ierr2)
ierr = ierr + ierr2
call kim_model_driver_create_set_destroy_pointer( &
  model_driver_create_handle, kim_language_name_fortran, &
  c_funloc(model_destroy), ierr2)
ierr = ierr + ierr2
if (ierr /= 0) then
  kim_log_message = "Unable to store callback pointers"
  LOG_ERROR()
  return
end if

! process parameter file
call kim_model_driver_create_get_number_of_parameter_files( &
  model_driver_create_handle, number_of_parameter_files)
if (number_of_parameter_files .ne. 1) then
  kim_log_message = "Wrong number of parameter files"
  ierr = 1
  LOG_ERROR()
  return
end if

! allocate model_buffer object and register it in the model_drier_create object
allocate(buffer)
call kim_model_driver_create_set_model_buffer_pointer( &
  model_driver_create_handle, c_loc(buffer))

! Read in model parameters from parameter file
!
call kim_model_driver_create_get_parameter_file_name( &
  model_driver_create_handle, 1, parameter_file_name, ierr)
if (ierr /= 0) then
  kim_log_message = "Unable to get parameter file name"
  LOG_ERROR()
  return
end if
open(10,file=parameter_file_name,status="old")
read(10,*,iostat=ierr,err=100) buffer%num_species
allocate(buffer%species_name_strings(buffer%num_species))
read(10,*,iostat=ierr,err=100) (buffer%species_name_strings(p), p = 1, buffer%num_species)
allocate(buffer%num_symmetry_functions_of_species(buffer%num_species))
read(10,*,iostat=ierr,err=100) (buffer%num_symmetry_functions_of_species(p), p = 1, buffer%num_species)
allocate(buffer%symmetry_functions(buffer%num_species))
do p = 1, buffer%num_species
  buffer%symmetry_functions(p)%center_atom_species_code = p
  buffer%symmetry_functions(p)%center_atom_species = buffer%species_name_strings(p)
  allocate(buffer%symmetry_functions(p)%gs(buffer%num_symmetry_functions_of_species(p)))
  do q = 1, buffer%num_symmetry_functions_of_species(p)
    read(10,*,iostat=ierr,err=100) &
    junk, &
    buffer%symmetry_functions(p)%gs(q)%symmetry_type
    if (buffer%symmetry_functions(p)%gs(q)%symmetry_type == 'g2') then
      read(10,*,iostat=ierr,err=100) &
      buffer%symmetry_functions(p)%gs(q)%species1_name_string, &
      buffer%symmetry_functions(p)%gs(q)%eta
      do k = 1, buffer%num_species
        if (buffer%species_name_strings(k) == buffer%symmetry_functions(p)%gs(q)%species1_name_string) then
          exit
        end if
      end do
      buffer%symmetry_functions(p)%gs(q)%species1_code = k
      call kim_species_name_from_string(trim(buffer%symmetry_functions(p)%gs(q)%species1_name_string), &
      buffer%symmetry_functions(p)%gs(q)%species1_name)
      call kim_model_driver_create_set_species_code( &
      model_driver_create_handle, buffer%symmetry_functions(p)%gs(q)%species1_name, &
      buffer%symmetry_functions(p)%gs(q)%species1_code, ierr)
    else if (buffer%symmetry_functions(p)%gs(q)%symmetry_type == 'g4') then
      read(10,*,iostat=ierr,err=100) &
      buffer%symmetry_functions(p)%gs(q)%species1_name_string, &
      buffer%symmetry_functions(p)%gs(q)%species2_name_string, &
      buffer%symmetry_functions(p)%gs(q)%eta, &
      buffer%symmetry_functions(p)%gs(q)%gamma, &
      buffer%symmetry_functions(p)%gs(q)%zeta
      do k = 1, buffer%num_species
        if (buffer%species_name_strings(k) == buffer%symmetry_functions(p)%gs(q)%species1_name_string) then
          exit
        end if
      end do
      buffer%symmetry_functions(p)%gs(q)%species1_code = k
      call kim_species_name_from_string(trim(buffer%symmetry_functions(p)%gs(q)%species1_name_string), &
      buffer%symmetry_functions(p)%gs(q)%species1_name)
      call kim_model_driver_create_set_species_code( &
      model_driver_create_handle, buffer%symmetry_functions(p)%gs(q)%species1_name, &
      buffer%symmetry_functions(p)%gs(q)%species1_code, ierr)
      do k = 1, buffer%num_species
        if (buffer%species_name_strings(k) == buffer%symmetry_functions(p)%gs(q)%species2_name_string) then
          exit
        end if
      end do
      buffer%symmetry_functions(p)%gs(q)%species2_code = k
      call kim_species_name_from_string(trim(buffer%symmetry_functions(p)%gs(q)%species2_name_string), &
      buffer%symmetry_functions(p)%gs(q)%species2_name)
      call kim_model_driver_create_set_species_code( &
      model_driver_create_handle, buffer%symmetry_functions(p)%gs(q)%species2_name, &
      buffer%symmetry_functions(p)%gs(q)%species2_code, ierr)
    end if
    read(10,*,iostat=ierr,err=100) buffer%symmetry_functions(p)%gs(q)%min_fingerprint, &
    buffer%symmetry_functions(p)%gs(q)%max_fingerprint
  end do
end do
read(10,*,iostat=ierr,err=100) buffer%cutoff(1)  ! in A
allocate(buffer%model_props(buffer%num_species))
read(10,*,iostat=ierr,err=100) buffer%model_props(1)%activation_function
do p = 2, buffer%num_species
  buffer%model_props(p)%activation_function = buffer%model_props(1)%activation_function
end do

do p = 1, buffer%num_species
  read(10,*,iostat=ierr,err=100) buffer%model_props(p)%no_hiddenlayers_of_species
  allocate(buffer%model_props(p)%no_nodes_of_species(buffer%model_props(p)%no_hiddenlayers_of_species))
  read(10,*,iostat=ierr,err=100) &
  (buffer%model_props(p)%no_nodes_of_species(q), q = 1, buffer%model_props(p)%no_hiddenlayers_of_species)
end do

count = 0
do p = 1, buffer%num_species
  if (buffer%model_props(p)%no_hiddenlayers_of_species == 0) then
    count = count + (size(buffer%symmetry_functions(p)%gs) + 1)
  else
    count = count + (size(buffer%symmetry_functions(p)%gs) + 1) * buffer%model_props(p)%no_nodes_of_species(1)
    do q = 1, buffer%model_props(p)%no_hiddenlayers_of_species - 1
      count = count + (buffer%model_props(p)%no_nodes_of_species(q) + 1) * buffer%model_props(p)%no_nodes_of_species(q + 1)
    end do
    count = count + (buffer%model_props(p)%no_nodes_of_species(buffer%model_props(p)%no_hiddenlayers_of_species) + 1)
  end if
  count = count + 2
end do
allocate(buffer%parameters(count))
read(10,*,iostat=ierr,err=100) (buffer%parameters(p), p = 1, count)
close(10)
goto 200
100 continue
! reading parameters failed
kim_log_message = "Unable to read parameters"
ierr = 1
LOG_ERROR()
return
200 continue

! register species
allocate(buffer%species_names(buffer%num_species))
allocate(buffer%species_codes(buffer%num_species))
do p = 1, buffer%num_species
  call kim_species_name_from_string(trim(buffer%species_name_strings(p)), &
  buffer%species_names(p))
  buffer%species_codes(p) = p
  call kim_model_driver_create_set_species_code( &
  model_driver_create_handle, buffer%species_names(p), buffer%species_codes(p), ierr)
end do
if (ierr /= 0) then
  kim_log_message = "Unable to set species code"
  LOG_ERROR()
  return
end if

! convert units of parameters
call kim_model_driver_create_convert_unit( &
  model_driver_create_handle, &
  kim_length_unit_a, &
  kim_energy_unit_unused, &
  kim_charge_unit_unused, &
  kim_temperature_unit_unused, &
  kim_time_unit_unused, &
  requested_length_unit, &
  requested_energy_unit, &
  requested_charge_unit, &
  requested_temperature_unit, &
  requested_time_unit, &
  1.0_cd, 0.0_cd, 0.0_cd, 0.0_cd, 0.0_cd, unit_conversion_factor, ierr)
if (ierr /= 0) then
  kim_log_message = "Unable to convert length unit"
  LOG_ERROR()
  return
end if
buffer%cutoff(1) = unit_conversion_factor * buffer%cutoff(1)

call kim_model_driver_create_convert_unit( &
  model_driver_create_handle, &
  kim_length_unit_unused, &
  kim_energy_unit_ev, &
  kim_charge_unit_unused, &
  kim_temperature_unit_unused, &
  kim_time_unit_unused, &
  requested_length_unit, &
  requested_energy_unit, &
  requested_charge_unit, &
  requested_temperature_unit, &
  requested_time_unit, &
  0.0_cd, 1.0_cd, 0.0_cd, 0.0_cd, 0.0_cd, unit_conversion_factor, ierr)
if (ierr /= 0) then
  kim_log_message = "Unable to convert energy unit"
  LOG_ERROR()
  return
end if
!buffer%hyperparameter = unit_conversion_factor * buffer%hyperparameter

! Set remainder of parameters
buffer%influence_distance = buffer%cutoff(1)
buffer%padding_neighbor_hints(1) = 1
buffer%half_list_hints(1) = 1

! register influence distance
call kim_model_driver_create_set_influence_distance_pointer( &
  model_driver_create_handle, buffer%influence_distance)

! register cutoff
call kim_model_driver_create_set_neighbor_list_pointers( &
  model_driver_create_handle, 1, buffer%cutoff, &
  buffer%padding_neighbor_hints)
  !, buffer%half_list_hints)

ierr = 0  ! everything is good
end subroutine model_driver_create
