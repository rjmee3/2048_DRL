cmake_minimum_required(VERSION 3.22.1)
project(2048_DRL)

# Add the main executable
add_executable(2048_DRL driver.c)

# link libraries
target_link_libraries(2048_DRL m)

# Add the other source files to the executable
target_sources(2048_DRL PRIVATE GameState.c Queue.c NeuralNetwork.c)

# Include directories for header files
target_include_directories(2048_DRL PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
