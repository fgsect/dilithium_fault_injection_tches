#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "randombytes.h"

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#else
#include <fcntl.h>
#include <errno.h>
#ifdef __linux__
#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#else
#include <unistd.h>
#endif
#endif

#define STR_(X) #X
#define STR(X) STR_(X)

uint8_t pseudorandombytes_yesplease = 0;
keccak_state pseudorandombytes_state;

void pseudorandombytes_seed(uint8_t *seed, size_t seedlen) {
  if (seedlen) {
    shake256_init(&pseudorandombytes_state);
    shake256_absorb(&pseudorandombytes_state, seed, seedlen);
    shake256_finalize(&pseudorandombytes_state);

    pseudorandombytes_yesplease = 1;
  } else {
    pseudorandombytes_yesplease = 0;
  }
}

#ifdef PSEUDORANDOMBYTES_SEED
void randombytes(uint8_t *out, size_t outlen) {
  if (!pseudorandombytes_yesplease) {
    pseudorandombytes_seed(STR(PSEUDORANDOMBYTES_SEED), sizeof(STR(PSEUDORANDOMBYTES_SEED)) - 1)
  }
  shake256_squeeze(out, outlen, &pseudorandombytes_state);
}
#elif defined( _WIN32)
void randombytes(uint8_t *out, size_t outlen) {
  if (pseudorandombytes_yesplease) {
    shake256_squeeze(out, outlen, &pseudorandombytes_state);
  } else {
    HCRYPTPROV ctx;
    size_t len;

    if(!CryptAcquireContext(&ctx, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT))
      abort();

    while(outlen > 0) {
      len = (outlen > 1048576) ? 1048576 : outlen;
      if(!CryptGenRandom(ctx, len, (BYTE *)out))
        abort();

      out += len;
      outlen -= len;
    }

    if(!CryptReleaseContext(ctx, 0))
      abort();
  }
}
#elif defined(__linux__) && defined(SYS_getrandom)
void randombytes(uint8_t *out, size_t outlen) {
  if (pseudorandombytes_yesplease) {
    shake256_squeeze(out, outlen, &pseudorandombytes_state);
  } else {
    ssize_t ret;

    while(outlen > 0) {
      ret = syscall(SYS_getrandom, out, outlen, 0);
      if(ret == -1 && errno == EINTR)
        continue;
      else if(ret == -1)
        abort();

      out += ret;
      outlen -= ret;
    }
  }
}
#else
void randombytes(uint8_t *out, size_t outlen) {
  if (pseudorandombytes_yesplease) {
    shake256_squeeze(out, outlen, &pseudorandombytes_state);
  } else {
    static int fd = -1;
    ssize_t ret;

    while(fd == -1) {
      fd = open("/dev/urandom", O_RDONLY);
      if(fd == -1 && errno == EINTR)
        continue;
      else if(fd == -1)
        abort();
    }

    while(outlen > 0) {
      ret = read(fd, out, outlen);
      if(ret == -1 && errno == EINTR)
        continue;
      else if(ret == -1)
        abort();

      out += ret;
      outlen -= ret;
    }
  }
}
#endif
