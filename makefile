# the compiler: g++
  CC = g++

  # compiler flags:
  CFLAGS  = -std=c++20 -fconcepts

  # the build target executable:
  TARGET = test

  all: $(TARGET)

  $(TARGET): $(TARGET).cpp
  	
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp

  clean:
  	
	$(RM) $(TARGET)
