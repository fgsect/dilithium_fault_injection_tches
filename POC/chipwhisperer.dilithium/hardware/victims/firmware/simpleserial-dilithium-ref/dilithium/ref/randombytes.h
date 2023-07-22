#ifndef RANDOMBYTES_H
#define RANDOMBYTES_H

#define MAX_SEED_LEN (512)

#include <stddef.h>
#include <stdint.h>
#include "fips202.h"

void randombytes(uint8_t *out, size_t outlen);
void pseudorandombytes_seed(uint8_t *seed, size_t seedlen);

#endif
