# Begin preamble

cmake_minimum_required(VERSION 3.4)

list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_V2_CMAKE_PREFIX_DIR})
find_package(KIM-API-V2 2.0 REQUIRED CONFIG)
if(NOT TARGET kim-api)
  project("${KIM_API_PROJECT_NAME}" VERSION "${KIM_API_VERSION}"
    LANGUAGES CXX C Fortran)
endif()

# End preamble

$(cp ../../amp/model/neuralnetwork.f90 amp_model_driver/neuralnetwork.f90)
$(cp ../../amp/descriptor/cutoffs.f90 amp_model_driver/cutoffs.f90)
$(cp ../../amp/descriptor/gaussian.f90 amp_model_driver/gaussian.f90)


set(MODEL_DRIVER_NAME "amp_model_driver")

add_kim_api_model_driver_library(
  NAME                    ${MODEL_DRIVER_NAME}
  CREATE_ROUTINE_NAME     "model_driver_create"
  CREATE_ROUTINE_LANGUAGE "fortran"
  )

target_sources(${MODEL_DRIVER_NAME} PRIVATE
  ${MODEL_DRIVER_NAME}.F90
  cutoffs.f90
  gaussian.f90
  neuralnetwork.f90
  )
