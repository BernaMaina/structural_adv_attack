CC = gcc
CFLAGS = -Wall -std=gnu99
CLIBS =

SDIR = src
ODIR = build

SRC = $(wildcard $(SDIR)/*.c)
OBJ = $(patsubst $(SDIR)/%.c, $(ODIR)/%.o, $(SRC))

DUMMY = dummy
PAYLOAD = payload
PAYLOAD_A = payload_a
PAYLOAD_B = payload_b
PAYLOAD_C = payload_c
PAYLOAD_D = payload_d

TARGET = elfinjector

.PHONY: all debug release clean dummy payload 

all: dummy payload debug payload_a payload_b payload_c payload_d

$(ODIR):
	mkdir -p $@

$(ODIR)/%.o: $(SDIR)/%.c
	$(CC) $(CFLAGS) $(CLIBS) -c $< -o $@

$(ODIR)/$(TARGET): $(ODIR) $(OBJ)
	$(CC) $(CFLAGS) $(CLIBS) -o $@ $(OBJ)

release: CFLAGS += -O3
release: $(ODIR)/$(TARGET) $(ODIR)

debug: CFLAGS += -g -DDEBUG
debug: $(ODIR)/$(TARGET) $(ODIR)

payload: $(PAYLOAD).s $(ODIR)
	nasm -f elf64 -O0 -o $(ODIR)/$(PAYLOAD).o $<
	ld -o $(ODIR)/$(PAYLOAD) $(ODIR)/$(PAYLOAD).o
    
payload_a: $(PAYLOAD_A).s $(ODIR)
	nasm -f elf64 -O0 -o $(ODIR)/$(PAYLOAD_A).o $<
	ld -o $(ODIR)/$(PAYLOAD_A) $(ODIR)/$(PAYLOAD_A).o

payload_b: $(PAYLOAD_B).s $(ODIR)
	nasm -f elf64 -O0 -o $(ODIR)/$(PAYLOAD_B).o $<
	ld -o $(ODIR)/$(PAYLOAD_B) $(ODIR)/$(PAYLOAD_B).o
    
payload_c: $(PAYLOAD_C).s $(ODIR)
	nasm -f elf64 -O0 -o $(ODIR)/$(PAYLOAD_C).o $<
	ld -o $(ODIR)/$(PAYLOAD_C) $(ODIR)/$(PAYLOAD_C).o
    
payload_d: $(PAYLOAD_D).s $(ODIR)
	nasm -f elf64 -O0 -o $(ODIR)/$(PAYLOAD_D).o $<
	ld -o $(ODIR)/$(PAYLOAD_D) $(ODIR)/$(PAYLOAD_D).o

dummy: $(DUMMY).c $(ODIR)
	$(CC) -O0 -no-pie -o $(ODIR)/$(DUMMY) $<

clean:
	rm -rf $(ODIR)/*

.SPECIAL: $(ODIR)/$(DUMMY)
