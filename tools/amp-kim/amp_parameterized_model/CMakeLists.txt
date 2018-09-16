# begin preamble

cmake_minimum_required(VERSION 3.4)

list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_V2_CMAKE_PREFIX_DIR})
find_package(KIM-API-V2 2.0 REQUIRED CONFIG)
if(NOT TARGET kim-api)
  project("${KIM_API_PROJECT_NAME}" VERSION "${KIM_API_VERSION}"
    LANGUAGES CXX C Fortran)
endif()

# End preamble


add_kim_api_model_library(
  NAME            "amp_parameterized_model"
  DRIVER_NAME     "amp_model_driver"
  PARAMETER_FILES "amp.params"
  )
