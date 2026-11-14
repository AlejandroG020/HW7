CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS += -lm

TARGET := verify_tester
OBJECTS := verify_tester.o mmul_sparse.o mmio.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -pthread -o $@ $(OBJECTS) $(LDLIBS)

verify_tester.o: matrix.h
mmul_sparse.o: matrix.h mmio.h
mmio.o: mmio.h

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
