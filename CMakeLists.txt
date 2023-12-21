cmake_minimum_required(VERSION 3.10)
project(MyProgram)

# Add the main executable
add_executable(my_program driver.c)

# Add the other source files to the executable
target_sources(my_program PRIVATE GameState.c Queue.c)

# Include directories for header files
target_include_directories(my_program PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optionally, you can set additional compiler flags
# target_compile_options(my_program PRIVATE -Wall -Wextra)
