/*
    This file is part of the ChipWhisperer Example Targets
    Copyright (C) 2012-2017 NewAE Technology Inc.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "simpleserial-dilithium-ref.h"

#include "hal.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "simpleserial.h"

#include "dilithium/ref/api.h"
#include "dilithium/ref/params.h"
#include "dilithium/ref/randombytes.h"
#include "dilithium/ref/poly.h"
#include "dilithium/ref/polyvec.h"

#define ENABLE_SIGNATURE
//#define ENABLE_POLYZ_UNPACK

#ifdef ENABLE_SIGNATURE
uint8_t secret_key[pqcrystals_dilithium5_SECRETKEYBYTES] = DEFAULT_SECRET_KEY;
uint8_t sig[CRYPTO_BYTES];
#endif // ENABLE_SIGNATURE

#ifdef ENABLE_POLYZ_UNPACK
uint8_t poly_packed[] = POLY_PACKED;
poly poly_unpacked;
#endif // ENABLE_POLYZ_UNPACK



#define ASSERT(cond, msg) do \
{ \
  if (!(cond)) { \
    simpleserial_put('a', sizeof(msg) - 1, (msg)); \
    return ASSERT_FAILED; \
  } \
} while (0)

#ifdef ENABLE_SIGNATURE
uint8_t sign(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf) {
  size_t siglen;

  ASSERT(cmd == CMD_SIGN, "sign: invalid cmd");
  ASSERT(scmd == 0, "sign: invalid scmd");

  int result = pqcrystals_dilithium2_ref_signature(sig, &siglen, buf, len, secret_key);

  ASSERT(siglen == pqcrystals_dilithium2_BYTES, "sign: signature has unexpected length");
  simpleserial_put('r', sizeof("sign ok") - 1, "sign ok");
  return result; // 0 == success
}

uint8_t get_sig(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf) {
  size_t num_packets;
  size_t last_packet_len;

  if (CRYPTO_BYTES % MAX_PAYLOAD_LENGTH) { // does not divide
    num_packets = CRYPTO_BYTES / MAX_PAYLOAD_LENGTH + 1;
    last_packet_len = CRYPTO_BYTES % MAX_PAYLOAD_LENGTH;
  } else { // does divide
    num_packets = CRYPTO_BYTES / MAX_PAYLOAD_LENGTH;
    last_packet_len = MAX_PAYLOAD_LENGTH;
  }

  ASSERT(scmd < num_packets, "get_sig: scmd out of range"); // prevent buffer overflow

  if (scmd == num_packets - 1) { // last packet
    simpleserial_put('r', last_packet_len, sig + scmd * MAX_PAYLOAD_LENGTH);
    return 0x00;
  }
  // not last packet; but valid scmd as of previous assert
  simpleserial_put('r', MAX_PAYLOAD_LENGTH, sig + scmd * MAX_PAYLOAD_LENGTH);
  return 0x00;
}
#endif // ENABLE_SIGNATURE

#ifdef ENABLE_POLYZ_UNPACK
uint8_t loop(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf) {
    ASSERT(cmd == CMD_LOOP, "loop: cmd");
    ASSERT(scmd == 0, "loop: scmd");
    ASSERT(len == 0, "loop: len");

    // mem is assumed to be zero
    memset(&poly_unpacked, 0, sizeof(poly_unpacked));

    polyz_unpack(&poly_unpacked, poly_packed); // secret key should be a big enough deterministic buffer I assume ...
    // pack again for faster transmission
    polyz_pack(poly_packed, &poly_unpacked);

  simpleserial_put('r', sizeof("loop ok") - 1, "loop ok");

    return 0x00;
}

uint8_t get_poly(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf) {
  size_t num_packets;
  size_t last_packet_len;

  if (POLYZ_PACKEDBYTES % MAX_PAYLOAD_LENGTH) { // does not divide
    num_packets = POLYZ_PACKEDBYTES / MAX_PAYLOAD_LENGTH + 1;
    last_packet_len = POLYZ_PACKEDBYTES % MAX_PAYLOAD_LENGTH;
  } else { // does divide
    num_packets = POLYZ_PACKEDBYTES / MAX_PAYLOAD_LENGTH;
    last_packet_len = MAX_PAYLOAD_LENGTH;
  }

  ASSERT(scmd < num_packets, "get_poly: scmd out of range"); // prevent buffer overflow

  if (scmd == num_packets - 1) { // last packet
    simpleserial_put('r', last_packet_len, poly_packed + scmd * MAX_PAYLOAD_LENGTH);
    return 0x00;
  }
  // not last packet; but valid scmd as of previous assert
  simpleserial_put('r', MAX_PAYLOAD_LENGTH, poly_packed + scmd * MAX_PAYLOAD_LENGTH);
  return 0x00;
}
#endif // ENABLE_POLYZ_UNPACK

int main(void)
{
  platform_init();
  init_uart();
  trigger_setup();

  simpleserial_init();

#ifdef ENABLE_SIGNATURE
  simpleserial_addcmd(CMD_SIGN, MAX_PAYLOAD_LENGTH, sign);
  simpleserial_addcmd(CMD_GET_SIG, MAX_PAYLOAD_LENGTH, get_sig);
#endif // ENABLE_SIGNATURE

#ifdef ENABLE_POLYZ_UNPACK
  simpleserial_addcmd(CMD_LOOP, 0, loop);
  simpleserial_addcmd(CMD_GET_POLY, 0, get_poly);
#endif // ENABLE_POLYZ_UNPACK

  // signal up and running
  simpleserial_put('b', sizeof("boot ok") - 1, "boot ok");
  while(1)
    simpleserial_get();
}
