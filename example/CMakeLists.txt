set(PROGS openmp)
foreach(name ${PROGS})
  set(target "example_${name}")
  add_executable(${target} ${name}.cpp)
  set_target_properties(${target} PROPERTIES OUTPUT_NAME ${name})
  target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
endforeach(name)

if(MPI_FOUND)
  set(PROGS hybrid mpi)
  foreach(name ${PROGS})
    set(target "example_${name}")
    add_executable(${target} ${name}.cpp)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${name})
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)
  endforeach(name)
endif(MPI_FOUND)
