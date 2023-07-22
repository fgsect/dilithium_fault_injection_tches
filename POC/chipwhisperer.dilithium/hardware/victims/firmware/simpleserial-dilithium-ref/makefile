# Hey Emacs, this is a -*- makefile -*-
#----------------------------------------------------------------------------
#
# Makefile for ChipWhisperer SimpleSerial-AES Program
#
#----------------------------------------------------------------------------
# On command line:
#
# make all = Make software.
#
# make clean = Clean out built project files.
#
# make coff = Convert ELF to AVR COFF.
#
# make extcoff = Convert ELF to AVR Extended COFF.
#
# make program = Download the hex file to the device, using avrdude.
#                Please customize the avrdude settings below first!
#
# make debug = Start either simulavr or avarice as specified for debugging,
#              with avr-gdb or avr-insight as the front end for debugging.
#
# make filename.s = Just compile filename.c into the assembler code only.
#
# make filename.i = Create a preprocessed source file for use in submitting
#                   bug reports to the GCC project.
#
# To rebuild project do "make clean" then "make all".
#----------------------------------------------------------------------------

# Target file name (without extension).
# This is the name of the compiled .hex file.
TARGET = simpleserial-dilithium-ref

CDEFS = -DSHUFFLE=1

DILITHIUM_SOURCES = dilithium/ref/sign.c dilithium/ref/packing.c dilithium/ref/polyvec.c dilithium/ref/poly.c dilithium/ref/ntt.c dilithium/ref/reduce.c dilithium/ref/rounding.c
DILITHIUM_FIPS202_SRCS= $(DILITHIUM_SOURCES) dilithium/ref/symmetric-shake.c

# List C source files here.
# Header files (.h) are automatically pulled in.
SRC += simpleserial-dilithium-ref.c $(DILITHIUM_FIPS202_SRCS) dilithium/ref/fips202.c

# -----------------------------------------------------------------------------

SS_VER = SS_VER_2_1
CRYPTO_TARGET = NONE
CRYPTO_OPTIONS = NONE

ifeq ($(PLATFORM),)
PLATFORM = CW308_STM32F4
endif

#Add simpleserial project to build
include ../simpleserial/Makefile.simpleserial

FIRMWAREPATH = ../.
include $(FIRMWAREPATH)/Makefile.inc
