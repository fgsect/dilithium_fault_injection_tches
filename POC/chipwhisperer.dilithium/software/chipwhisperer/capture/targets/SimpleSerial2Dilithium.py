import functools

import chipwhisperer.capture.targets.SimpleSerial2 as SimpleSerial2
import struct
import importlib
import time
from typing import List, Optional, Dict
import numpy as np
import logging
import math
from dilithium import _params as dilithium_params # is "python-dilithium" in path?
from dilithium import Dilithium # is "python-dilithium" in path?

class TargetIOError(BlockingIOError):
    @property
    def data(self):
        return self.__data

    def __init__(self, message: str, data):
        super().__init__(message)
        self.__data = data

class TargetTimeoutError(TargetIOError):
    def __init__(self, message: str = None):
        if message is None:
            message = 'Target cleanly timed out while generating a signature'
        super().__init__(message, b'')


class LogToExceptionHandler(logging.NullHandler):
    def __init__():
        super().__init__()
        self.setLevel(logging.NOTSET)

    @property
    def warning_or_higher_logged(self) -> bool:
        return self.__warning_or_higher_logged

    @property
    def records_warning_or_higher(self) -> list:
        return [record for record in self.__records_warning_or_higher]

    def reset(self) -> None:
        self.__warning_or_higher_logged = False
        self.__records_warning_or_higher = []

    def __init__(self):
        self.__warning_or_higher_logged = False
        self.__records_warning_or_higher = []

    def handle(self, record):
        if record.levelno >= logging.WARNING:
            self.__warning_or_higher_logged = True
            self.__records_warning_or_higher += [record]

class SimpleSerial2Dilithium(SimpleSerial2):
    __handler = LogToExceptionHandler()
    __ALGORITHMS = [2, 3, 5]
    __MAX_PAYLOAD_LENGTH = 128 #64 #120 #128 #249

    __COMMAND_ALGORITHM = 'q'
    __COMMAND_SET_SECRET_KEY = 'k'
    __COMMAND_SIGN = 'e'
    __COMMAND_GET_SIGN = 'g'
    __COMMAND_LOOP = 'l'
    __COMMAND_GET_POLY = 'n'
    __FIRST_ERR_RATE_PAYLOAD249_ITER100 = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 3., 3., 19., 12., 11., 7., 10., 5., 5., 9., 2., 4., 2., 1., 0., 1., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    __FIRST_ERROR_RATE_PAYLOAD128_ITER10000 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    __FIRST_ERROR_RATE_PAYLOAD128_ITER100000 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 2, 0, 1, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]
    # 100 Dilithium (2) signature timings in seconds of messages (0x0000 - 0x0064)
    __SIG_TIMINGS = [1.8345985412597656, 0.571202278137207, 0.5727174282073975, 2.44435453414917, 1.9954373836517334, 1.956618309020996, 1.0284478664398193, 0.5720925331115723, 1.0222575664520264, 0.5756478309631348, 3.327014446258545, 1.1550006866455078, 3.5656371116638184, 1.4125359058380127, 1.2258155345916748, 0.5754551887512207, 0.5706188678741455, 2.6741480827331543, 1.2207884788513184, 0.569699764251709, 1.8295707702636719, 2.4150185585021973, 2.5315682888031006, 1.6380705833435059, 0.5689265727996826, 0.7610783576965332, 0.800501823425293, 2.897522211074829, 1.539381504058838, 1.148653507232666, 0.5714447498321533, 2.256739377975464, 0.795245885848999, 0.956822395324707, 0.5665726661682129, 0.5721073150634766, 1.150632381439209, 1.2155184745788574, 1.6022050380706787, 0.9530470371246338, 0.7947635650634766, 1.220771312713623, 1.7636890411376953, 0.7999839782714844, 2.0266835689544678, 2.9747705459594727, 1.5745739936828613, 0.5741910934448242, 1.4473426342010498, 1.153580665588379, 0.5740790367126465, 0.7645726203918457, 0.5718286037445068, 3.671562433242798, 1.5743498802185059, 2.4453606605529785, 0.5700263977050781, 0.7969129085540771, 0.5690975189208984, 1.2237257957458496, 0.571495532989502, 0.9935479164123535, 0.7666583061218262, 1.0304381847381592, 0.8014285564422607, 0.5709078311920166, 0.7657623291015625, 0.7964329719543457, 1.832737922668457, 0.5714242458343506, 1.1860861778259277, 0.571995735168457, 0.7650182247161865, 1.0285441875457764, 2.029315710067749, 0.987626314163208, 0.7646205425262451, 2.2466177940368652, 1.6047735214233398, 0.7635841369628906, 0.5692150592803955, 0.7618019580841064, 1.6753244400024414, 0.9920835494995117, 0.5694153308868408, 0.7975795269012451, 1.4073009490966797, 2.9902737140655518, 0.5741786956787109, 0.8005082607269287, 2.445647716522217, 0.5705244541168213, 0.993781566619873, 0.7978324890136719, 3.0634548664093018, 0.7647135257720947, 0.7613015174865723, 1.8281018733978271, 0.5725083351135254, 0.993952751159668]

    __timeout_strings = [
        'reset pre',
        'rst high 0',
        'rst low 0',
        'rst high 1',
        'rst low 1',
        'rst high 2',
        'rst low 2',
        'rst high 3',
    ]

    @property
    def algorithm(self) -> int:
        return self.__algorithm

    @algorithm.setter
    def algorithm(self, a: int):
        if a not in self.__ALGORITHMS:
            raise ValueError()
        # self.send_cmd(self.__COMMAND_ALGORITHM, 0, struct.pack('B', a))
        # response = self.simpleserial_read(cmd='r')
        # assert response is not None
        # assert response.startswith(b'set_alg ok: ' + struct.pack('B', a))
        self.__algorithm = a

    @property
    def secret_key(self) -> bytes:
        return self.__secret_key

    @property
    def crypto_secretkeybytes(self) -> int:
        return dilithium_params[self.__algorithm]['CRYPTO_SECRETKEYBYTES']

    @property
    def crypto_bytes(self) -> int:
        return dilithium_params[self.__algorithm]['CRYPTO_BYTES']

    @property
    def polyz_packedbytes(self) -> int:
        return dilithium_params[self.__algorithm]['POLYZ_PACKEDBYTES']

    @property
    def dilithium(self):
        return self.__d
    
    @secret_key.setter
    def secret_key(self, s: bytes):
        if len(s) != self.crypto_secretkeybytes:
            raise ValueError(f'Expected secret key of length {self.crypto_bytes} but got {len(s)}')
        # self.send_cmd_long(self.__COMMAND_SECRET_KEY, s)
        self.__secre_key = s

    __iteration_duration_cache = None

    def dglitch_settings(self):
        """
        Dilithium glitch settings
        """
        self.scope.default_setup()
        self.scope.cglitch_setup()  # default_setup for clock glitching
        self.scope.clock.adc_src = "clkgen_x1"  # scope.adc.trig_count will be in units of clock cycles

    def run_without_glitch(self, action: callable):
        self.scope.io.hs2 = "clkgen"  # disable glitch
        self.scope.sc.arm(False)  # reset trig_count
        self.scope.arm()

        res = action()

        self.scope.io.hs2 = "glitch"  # enable glitch

        return res

    @property
    @functools.lru_cache(maxsize=None)  # replace with @functools.cache when switching to python3.9 or above
    def loop_duration(self) -> int:
        self.run_without_glitch(self.loop)
        return self.scope.adc.trig_count

    @property
    def loop_duration_threshold(self):
        # this constant is great enough that it filters almost all false positives but
        # also always allows faults even in the last poly_index
        return self.loop_duration - 30

    @property
    @functools.lru_cache(maxsize=None)  # replace with @functools.cache when switching to python3.9 or above
    def loop_duration_sign(self) -> int:
        self.run_without_glitch(functools.partial(self.sign, b'\x01'))  # b'\x00' is not rejected
        return math.ceil(self.scope.adc.trig_count // self.__d.l)

    @property
    def loop_duration_sign_threshold(self):
        # this constant is great enough that it filters almost all false positives but
        # also always allows faults even in the last poly_index
        return self.loop_duration_sign - 30

    @functools.lru_cache(maxsize=None)  # replace with @functools.cache when switching to python3.9 or above
    def message_without_rejections(self, poly_index: Optional[int] = None) -> (bytes, bytes):
        """Returns message, signature_packed"""
        if poly_index is None:
            poly_index = 1000  # this index is out of range thus it will not fault
        message = None
        signature_packed = None
        for i in range(2 ** 16 - 1):
            upper = i // 256
            lower = i % 256
            message = bytes([upper, lower])
            signature_packed, num_rejections = self.__d.signature_faulted(message, self.secret_key, 0, poly_index)
            if num_rejections != 0:
                message = None
                continue
            else:
                break
        if message is None or signature_packed is None:
            raise RuntimeException(
                'We did not find a message without rejections searching two full bytes. While theoretically possible, it is more likely that the "signature_faulted" implementation is wrong.')
        return message, signature_packed

    @property
    @functools.lru_cache(maxsize=None)  # replace with @functools.cache when switching to python3.9 or above
    def signature_predictions(self) -> Dict:
        signatures_map = {}
        for poly_index in range(0, self.__d._polyz_unpack_num_iters):  # this loops includes a non-faulted signature
            message, signature_packed = self.message_without_rejections(poly_index)
            signatures_map[signature_packed] = message
        return signatures_map

    @property
    @functools.lru_cache(maxsize=None)  # replace with @functools.cache when switching to python3.9 or above
    def poly_no_fault(self) -> np.ndarray:
        def loop_and_get() -> bytes:
            self.loop()
            return self.get_poly()
        poly_packed = self.run_without_glitch(loop_and_get)
        return self.__d._polyz_unpack(poly_packed)

    @property
    @functools.lru_cache(maxsize=None)  # replace with @functools.cache when switching to python3.9 or above
    def poly_predictions(self) -> Dict:
        polys_map = {}
        for poly_index in range(0, self.__d._polyz_unpack_num_iters):
            split_index = (poly_index + 1) * self.__d._polyz_unpack_coeffs_per_iter
            poly_faulted = np.concatenate((self.poly_no_fault[:split_index], self.poly_no_fault[split_index:]))
            poly_faulted_packed = self.__d._polyz_pack(poly_faulted)
            polys_map[poly_faulted_packed] = poly_index
        return polys_map

    @property
    def iteration_duration(self) -> int:
        """
        This is an upper bound!
        """
        if self.__iteration_duration_cache is None:
            self.scope.glitch_disable()
            self.scope.arm()
            self.loop()
            timeout = self.scope.capture()
            assert not timeout
            self.__iteration_duration_cache = math.ceil(self.scope.adc.trig_count / self.__d._polyz_unpack_num_iters)
            self.dglitch_settings()
        return self.__iteration_duration_cache

    @property
    def all_possible_offsets(self) -> [int]:
        cache = [-49.609375, -49.21875, -48.828125, -48.4375, -48.046875, -47.65625, -47.265625, -46.875, -46.484375, -46.09375, -45.703125, -45.3125, -44.921875, -44.53125, -44.140625, -43.75, -43.359375, -42.96875, -42.578125, -42.1875, -41.796875, -41.40625, -41.015625, -40.625, -40.234375, -39.84375, -39.453125, -39.0625, -38.671875, -38.28125, -37.890625, -37.5, -37.109375, -36.71875, -36.328125, -35.9375, -35.546875, -35.15625, -34.765625, -34.375, -33.984375, -33.59375, -33.203125, -32.8125, -32.421875, -32.03125, -31.640625, -31.25, -30.859375, -30.46875, -30.078125, -29.6875, -29.296875, -28.90625, -28.515625, -28.125, -27.734375, -27.34375, -26.953125, -26.5625, -26.171875, -25.78125, -25.390625, -25.0, -24.609375, -24.21875, -23.828125, -23.4375, -23.046875, -22.65625, -22.265625, -21.875, -21.484375, -21.09375, -20.703125, -20.3125, -19.921875, -19.53125, -19.140625, -18.75, -18.359375, -17.96875, -17.578125, -17.1875, -16.796875, -16.40625, -16.015625, -15.625, -15.234375, -14.84375, -14.453125, -14.0625, -13.671875, -13.28125, -12.890625, -12.5, -12.109375, -11.71875, -11.328125, -10.9375, -10.546875, -10.15625, -9.765625, -9.375, -8.984375, -8.59375, -8.203125, -7.8125, -7.421875, -7.03125, -6.640625, -6.25, -5.859375, -5.46875, -5.078125, -4.6875, -4.296875, -3.90625, -3.515625, -3.125, -2.734375, -2.34375, -1.953125, -1.5625, -1.171875, -0.78125, -0.390625, 0.0, 0.390625, 0.78125, 1.171875, 1.5625, 1.953125, 2.34375, 2.734375, 3.125, 3.515625, 3.90625, 4.296875, 4.6875, 5.078125, 5.46875, 5.859375, 6.25, 6.640625, 7.03125, 7.421875, 7.8125, 8.203125, 8.59375, 8.984375, 9.375, 9.765625, 10.15625, 10.546875, 10.9375, 11.328125, 11.71875, 12.109375, 12.5, 12.890625, 13.28125, 13.671875, 14.0625, 14.453125, 14.84375, 15.234375, 15.625, 16.015625, 16.40625, 16.796875, 17.1875, 17.578125, 17.96875, 18.359375, 18.75, 19.140625, 19.53125, 19.921875, 20.3125, 20.703125, 21.09375, 21.484375, 21.875, 22.265625, 22.65625, 23.046875, 23.4375, 23.828125, 24.21875, 24.609375, 25.0, 25.390625, 25.78125, 26.171875, 26.5625, 26.953125, 27.34375, 27.734375, 28.125, 28.515625, 28.90625, 29.296875, 29.6875, 30.078125, 30.46875, 30.859375, 31.25, 31.640625, 32.03125, 32.421875, 32.8125, 33.203125, 33.59375, 33.984375, 34.375, 34.765625, 35.15625, 35.546875, 35.9375, 36.328125, 36.71875, 37.109375, 37.5, 37.890625, 38.28125, 38.671875, 39.0625, 39.453125, 39.84375, 40.234375, 40.625, 41.015625, 41.40625, 41.796875, 42.1875, 42.578125, 42.96875, 43.359375, 43.75, 44.140625, 44.53125, 44.921875, 45.3125, 45.703125, 46.09375, 46.484375, 46.875, 47.265625, 47.65625, 48.046875, 48.4375, 48.828125, 49.21875, 49.609375]
        if cache:
            return cache

        offsets_previous = set()
        offsets = set()
        step = 10
        while len(offsets) == 0 or len(offsets) > len(offsets_previous):
            offsets_previous = offsets
            offsets = set()
            for w in np.arange(self.scope.glitch.cwg._min_offset, self.scope.glitch.cwg._max_offset, step):
                self.scope.glitch.offset = w
                offsets.update({self.scope.glitch.offset})
            self.__logger.debug(f'all_possible_offsets: step: {step} -> {step / 10}')
            step /= 10

    @property
    def all_possible_widths(self) -> [int]:
        cache = [-49.609375, -49.21875, -48.828125, -48.4375, -48.046875, -47.65625, -47.265625, -46.875, -46.484375, -46.09375, -45.703125, -45.3125, -44.921875, -44.53125, -44.140625, -43.75, -43.359375, -42.96875, -42.578125, -42.1875, -41.796875, -41.40625, -41.015625, -40.625, -40.234375, -39.84375, -39.453125, -39.0625, -38.671875, -38.28125, -37.890625, -37.5, -37.109375, -36.71875, -36.328125, -35.9375, -35.546875, -35.15625, -34.765625, -34.375, -33.984375, -33.59375, -33.203125, -32.8125, -32.421875, -32.03125, -31.640625, -31.25, -30.859375, -30.46875, -30.078125, -29.6875, -29.296875, -28.90625, -28.515625, -28.125, -27.734375, -27.34375, -26.953125, -26.5625, -26.171875, -25.78125, -25.390625, -25.0, -24.609375, -24.21875, -23.828125, -23.4375, -23.046875, -22.65625, -22.265625, -21.875, -21.484375, -21.09375, -20.703125, -20.3125, -19.921875, -19.53125, -19.140625, -18.75, -18.359375, -17.96875, -17.578125, -17.1875, -16.796875, -16.40625, -16.015625, -15.625, -15.234375, -14.84375, -14.453125, -14.0625, -13.671875, -13.28125, -12.890625, -12.5, -12.109375, -11.71875, -11.328125, -10.9375, -10.546875, -10.15625, -9.765625, -9.375, -8.984375, -8.59375, -8.203125, -7.8125, -7.421875, -7.03125, -6.640625, -6.25, -5.859375, -5.46875, -5.078125, -4.6875, -4.296875, -3.90625, -3.515625, -3.125, -2.734375, -2.34375, -1.953125, -1.5625, -1.171875, -0.78125, -0.390625, 0.0, 0.390625, 0.78125, 1.171875, 1.5625, 1.953125, 2.34375, 2.734375, 3.125, 3.515625, 3.90625, 4.296875, 4.6875, 5.078125, 5.46875, 5.859375, 6.25, 6.640625, 7.03125, 7.421875, 7.8125, 8.203125, 8.59375, 8.984375, 9.375, 9.765625, 10.15625, 10.546875, 10.9375, 11.328125, 11.71875, 12.109375, 12.5, 12.890625, 13.28125, 13.671875, 14.0625, 14.453125, 14.84375, 15.234375, 15.625, 16.015625, 16.40625, 16.796875, 17.1875, 17.578125, 17.96875, 18.359375, 18.75, 19.140625, 19.53125, 19.921875, 20.3125, 20.703125, 21.09375, 21.484375, 21.875, 22.265625, 22.65625, 23.046875, 23.4375, 23.828125, 24.21875, 24.609375, 25.0, 25.390625, 25.78125, 26.171875, 26.5625, 26.953125, 27.34375, 27.734375, 28.125, 28.515625, 28.90625, 29.296875, 29.6875, 30.078125, 30.46875, 30.859375, 31.25, 31.640625, 32.03125, 32.421875, 32.8125, 33.203125, 33.59375, 33.984375, 34.375, 34.765625, 35.15625, 35.546875, 35.9375, 36.328125, 36.71875, 37.109375, 37.5, 37.890625, 38.28125, 38.671875, 39.0625, 39.453125, 39.84375, 40.234375, 40.625, 41.015625, 41.40625, 41.796875, 42.1875, 42.578125, 42.96875, 43.359375, 43.75, 44.140625, 44.53125, 44.921875, 45.3125, 45.703125, 46.09375, 46.484375, 46.875, 47.265625, 47.65625, 48.046875, 48.4375, 48.828125, 49.21875, 49.609375]
        # Negative offsets <-45 may result in double glitches!
        cache = [-44.921875, -44.53125, -44.140625, -43.75, -43.359375, -42.96875, -42.578125, -42.1875, -41.796875, -41.40625, -41.015625, -40.625, -40.234375, -39.84375, -39.453125, -39.0625, -38.671875, -38.28125, -37.890625, -37.5, -37.109375, -36.71875, -36.328125, -35.9375, -35.546875, -35.15625, -34.765625, -34.375, -33.984375, -33.59375, -33.203125, -32.8125, -32.421875, -32.03125, -31.640625, -31.25, -30.859375, -30.46875, -30.078125, -29.6875, -29.296875, -28.90625, -28.515625, -28.125, -27.734375, -27.34375, -26.953125, -26.5625, -26.171875, -25.78125, -25.390625, -25.0, -24.609375, -24.21875, -23.828125, -23.4375, -23.046875, -22.65625, -22.265625, -21.875, -21.484375, -21.09375, -20.703125, -20.3125, -19.921875, -19.53125, -19.140625, -18.75, -18.359375, -17.96875, -17.578125, -17.1875, -16.796875, -16.40625, -16.015625, -15.625, -15.234375, -14.84375, -14.453125, -14.0625, -13.671875, -13.28125, -12.890625, -12.5, -12.109375, -11.71875, -11.328125, -10.9375, -10.546875, -10.15625, -9.765625, -9.375, -8.984375, -8.59375, -8.203125, -7.8125, -7.421875, -7.03125, -6.640625, -6.25, -5.859375, -5.46875, -5.078125, -4.6875, -4.296875, -3.90625, -3.515625, -3.125, -2.734375, -2.34375, -1.953125, -1.5625, -1.171875, -0.78125, -0.390625, 0.0, 0.390625, 0.78125, 1.171875, 1.5625, 1.953125, 2.34375, 2.734375, 3.125, 3.515625, 3.90625, 4.296875, 4.6875, 5.078125, 5.46875, 5.859375, 6.25, 6.640625, 7.03125, 7.421875, 7.8125, 8.203125, 8.59375, 8.984375, 9.375, 9.765625, 10.15625, 10.546875, 10.9375, 11.328125, 11.71875, 12.109375, 12.5, 12.890625, 13.28125, 13.671875, 14.0625, 14.453125, 14.84375, 15.234375, 15.625, 16.015625, 16.40625, 16.796875, 17.1875, 17.578125, 17.96875, 18.359375, 18.75, 19.140625, 19.53125, 19.921875, 20.3125, 20.703125, 21.09375, 21.484375, 21.875, 22.265625, 22.65625, 23.046875, 23.4375, 23.828125, 24.21875, 24.609375, 25.0, 25.390625, 25.78125, 26.171875, 26.5625, 26.953125, 27.34375, 27.734375, 28.125, 28.515625, 28.90625, 29.296875, 29.6875, 30.078125, 30.46875, 30.859375, 31.25, 31.640625, 32.03125, 32.421875, 32.8125, 33.203125, 33.59375, 33.984375, 34.375, 34.765625, 35.15625, 35.546875, 35.9375, 36.328125, 36.71875, 37.109375, 37.5, 37.890625, 38.28125, 38.671875, 39.0625, 39.453125, 39.84375, 40.234375, 40.625, 41.015625, 41.40625, 41.796875, 42.1875, 42.578125, 42.96875, 43.359375, 43.75, 44.140625, 44.53125, 44.921875, 45.3125, 45.703125, 46.09375, 46.484375, 46.875, 47.265625, 47.65625, 48.046875, 48.4375, 48.828125, 49.21875, 49.609375]
        if cache:
            return cache

        widths_previous = set()
        widths = set()
        step = 10
        while len(widths) == 0 or len(widths) > len(widths_previous):
            widths_previous = widths
            widths = set()
            for w in np.arange(self.scope.glitch.cwg._min_width, self.scope.glitch.cwg._max_width, step):
                self.scope.glitch.width = w
                widths.update({self.scope.glitch.width})
            self.__logger.debug(f'all_possible_widths: step: {step} -> {step / 10}')
            step /= 10
        return widths

    def offsets_which_include(self, lower_offset: float, upper_offset: float, nonzero: bool = True):
        return self.sublist_which_include(lower_offset, upper_offset, self.all_possible_offsets, nonzero=nonzero)

    def widths_which_include(self, lower_width: float, upper_width: float, nonzero: bool = True):
        return self.sublist_which_include(lower_width, upper_width, self.all_possible_widths, nonzero=nonzero)

    def sublist_which_include(self, lower_width: float, upper_width: float, data: set, nonzero: bool = True) -> [int]:
        lower_index = None
        upper_index = None

        all_possible_widths_sorted = sorted(list(self.all_possible_widths))

        if lower_width > upper_width:
            raise ValueError('lower must be greater or equal to upper')
        if lower_width < self.all_possible_widths[0] or upper_width > self.all_possible_widths[-1]:
            raise ValueError(f'lower and upper are not in range [{all_possible_widths_sorted[0]}, {all_possible_widths_sorted[-1]}]')

        for i, width in enumerate(all_possible_widths_sorted):
            if lower_index is None:
                if width == lower_width:
                    lower_index = i
                elif width > lower_width:
                    lower_index = max(i - 1, 0)
            if upper_index is None:
                if width == upper_width:
                    upper_index = i
                elif width > upper_width:
                    upper_index = upper_index = min(len(self.all_possible_widths), i)
            if lower_index is not None and upper_index is not None:
                break

        ret = self.all_possible_widths[lower_index: upper_index + 1]
        if nonzero:
            ret = list(filter(lambda x: x != 0, ret))
        return ret

    @property
    def scope(self):
        return self.__scope


    @scope.setter
    def scope(self, s):
        assert s is not None
        self.__scope = s

    def sign(self, message: bytes, timeout: int = 10000) -> None:
        self.sign_send(message)
        return self.sign_recv(timeout)

    def sign_recv(self, timeout: int = 10000):
        ok_reply = b'sign ok'
        reply = self.simpleserial_read('r', len(ok_reply), timeout=timeout)
        assert reply == ok_reply

    def sign_send(self, message: bytes) -> None:
        if len(message) > self.__MAX_PAYLOAD_LENGTH:
            raise ValueError()
        self.send_cmd(self.__COMMAND_SIGN, 0, message)

    @property
    def timeout_strings(self):
        return tuple(self.__timeout_strings)

    def timeout_index_to_str(self, index: int) -> str:
        """
        index returned from sign_no_rej
        raises a value error if index is invalid
        """
        if index - 1 not in range(len(self.timeout_strings)):
            raise ValueError('invalid timeout index')

        return self.timeout_strings[index - 1]

    def sign_no_rej(self, message: bytes, timeout: int = 10000) -> (float,):
        from chipwhisperer.capture.scopes._OpenADCInterface import STATUS_EXT_MASK, ADDR_STATUS

        # from a random measurement
        TIME_PRE = 16718143905996707 - 16718143597047553
        TIME_HIGH = 16718143906529010 - 16718143905996707
        TIME_LOW = 16718143919473271 - 16718143906529010

        TIMEOUT_PRE_NS = TIME_PRE + round(.8 * TIME_LOW)
        TIMEOUT_HIGH_NS = TIME_HIGH * 5  # low is waaay lomger than high
        TIMEOUT_LOW_NS = TIME_LOW + round(.8 * TIME_LOW)


        handle = self.scope.adc.oa.serial.usbtx.handle  # directly work on libusb1 python package level
        dat_status_out = bytearray([1, 0, 0, 0, 2, 0, 0, 0])

        state_previous = False
        assert not state_previous  # trigger should be low

        gen_y_timeout_ns = 500e6  # half a second
        i_break = 2 * self.__d.l + 1

        i = 1
        delta = 1  # if it is initialitzed is it faster?
        timings = list(-1 for _ in range(self.__d.l * 2 + 1))
        counts = list(0 for _ in range(self.__d.l * 2 + 1))
        trig_counts = list(-1 for _ in range(self.__d.l * 2 + 1))

        self.scope.sc.arm(False)
        self.scope.arm()
        assert not self.scope.adc.state, "state should be low in the beginning"
        timings[0] = time.perf_counter_ns()
        self.sign_send(message)  # from here on time critical
        while True:
            # state_new = self.scope.adc.oa.serial.cmdReadMem(ADDR_STATUS, 1)[0] & STATUS_EXT_MASK  # == self.scope.adc.state
            handle.controlWrite(0x41, 18, 0, 0, dat_status_out, timeout=20000)
            state_new = handle.controlRead(0xC1, 18, 0, 0, 1, timeout=20000)[0] & STATUS_EXT_MASK
            # state_new = scope.adc.state
            counts[i] += 1
            if state_new != state_previous:  # edge
                timings[i] = time.perf_counter_ns()
                if not state_new:  # falling edge, now trig_count should be valid according to documentation
                    # while low we have way more time compared to being high
                    trig_counts[i] = self.scope.adc.trig_count  # inefficent, but sufficient?
                    self.scope.sc.arm(False)  # resets trig_count
                    self.scope.io.hs2 = "clkgen"  # disable glitch for the following poly samples
                    self.scope.arm()  # activates trig_count
                i += 1
            else:  # check for timeout
                delta = time.perf_counter_ns() - timings[i - 1]
                if i == 1 and delta > TIMEOUT_PRE_NS:
                    # could be that we missed high because of fault, better log last trig_count
                    trig_counts[2] = self.scope.adc.trig_count
                    break
                if i != 1 and i % 2 == 0 and delta > TIMEOUT_HIGH_NS:  # even means high
                    break
                if i != 1 and i % 2 == 1 and delta > TIMEOUT_LOW_NS:  # low
                    # could be that we missed high because of fault, better log last trig_count
                    trig_counts[i] = self.scope.adc.trig_count
                    break
            if i == i_break:  # check if done
                # not time critical anymore
                break

            state_previous = state_new
        # not time critical anymore

        # trim arrays
        counts = counts[1:]
        trig_counts = trig_counts[2::2]

        self.scope.io.hs2 = "glitch"  # enable glitches again for the next run

        self.__logger.warning(f'timings: {timings}')
        self.__logger.warning(f'counts: {counts}')
        self.__logger.warning(f'trig_counts: {trig_counts}')

        self.sign_recv(timeout)
        return timings, counts, trig_counts, None if i == i_break else self.timeout_index_to_str(i)

    def loop_send(self) -> None:
        self.send_cmd(self.__COMMAND_LOOP, 0, b'')

    def loop_recv(self, timeout=10) -> None:
        ok_reply = b'loop ok'
        reply = self.simpleserial_read('r', len(ok_reply), timeout=timeout)
        if reply != ok_reply:
            raise TargetIOError(f'Did not receive expected reply "{ok_reply} but instead got {reply}."', reply)

    def loop(self, timeout: int = 100) -> None:
        self.loop_send()
        self.loop_recv(timeout=timeout)

    def get_poly(self, max_num_retries: int = None) -> bytes:
        num_packets = math.ceil(self.polyz_packedbytes / self.__MAX_PAYLOAD_LENGTH)
        len_last_packet = self.polyz_packedbytes % self.__MAX_PAYLOAD_LENGTH if self.polyz_packedbytes % self.__MAX_PAYLOAD_LENGTH else self.__MAX_PAYLOAD_LENGTH
        dat = b''
        for i in range(num_packets - 1):
            dat += self.simpleserial_cmd_until_success(self.__COMMAND_GET_POLY, i, b'\xAA', cmd_read='r', pay_len=self.__MAX_PAYLOAD_LENGTH, max_num_retries=max_num_retries)
        dat += self.simpleserial_cmd_until_success(self.__COMMAND_GET_POLY, num_packets - 1, b'\xAA', cmd_read='r', pay_len=len_last_packet)
        return dat

    def get_sig(self) -> bytes:
        num_packets = math.ceil(self.crypto_bytes / self.__MAX_PAYLOAD_LENGTH)
        len_last_packet = self.crypto_bytes % self.__MAX_PAYLOAD_LENGTH if self.crypto_bytes % self.__MAX_PAYLOAD_LENGTH else self.__MAX_PAYLOAD_LENGTH
        dat = b''
        for i in range(num_packets - 1):
            dat += self.simpleserial_cmd_until_success(self.__COMMAND_GET_SIGN, i, b'\xAA', cmd_read='r', pay_len=self.__MAX_PAYLOAD_LENGTH)
        dat += self.simpleserial_cmd_until_success(self.__COMMAND_GET_SIGN, num_packets - 1, b'\xAA', cmd_read='r', pay_len=len_last_packet)
        return dat

    def read_until_blocking(self, pattern: bytes, timeout=1000) -> bytes:
        """yes, blocking; so that it is fast"""
        buf = b''
        start = time.time()
        while time.time() - start < timeout / 1000:
            onecharstring = self.read(num_char=1, timeout=1)
            assert type(onecharstring) is str
            assert len(onecharstring) in [0, 1]
            for char in onecharstring:
                buf += bytes([ord(char)]) # is that the correct way to convert that string? why is it even a string -,-
                if buf.endswith(pattern):
                    return buf
        raise TimeoutError(f'read_until_blocking timed out after {time.time() - start} s. (timeout={timeout / 1000}). Read until now: {buf}')

    def reboot_flush(self, timeout=2000):
        self.scope.io.nrst = False
        time.sleep(0.05)
        self.scope.io.nrst = "high_z"
        return self.wait_for_boot_msg(timeout)

    def wait_for_boot_msg(self, timeout, flush: bool = False):
        if flush:
            self.flush()
        # why is the overhead byte \x0b? Is it always that value? We will see ...
        data_read = self.read_until_blocking(b'\x0bb\x07boot ok\xc1\x00', timeout=timeout) # simpleserial_put('b', sizeof("boot ok") - 1, "boot ok");
        return data_read

    def check_error_rate(self, n: int) -> List[int]:
        assert n > 0
        expected = b'set_alg ok: 3' + 100 * b'Hello'
        expected = expected[:self.__MAX_PAYLOAD_LENGTH]

        def get_index_of_first_diff_bytes(a: bytes, b: bytes) -> int:
            if a == b:
                return None

            minlen = min(len(a), len(b))
            assert minlen >= 1
            for i in range(1, minlen):
                if a[:i] != b[:i]:
                    return i - 1

        dist = np.zeros(len(expected))
        for i in range(n):
            self.send_cmd(self.__COMMAND_ALGORITHM, 0, struct.pack('B', 3))
            try:
                response = self.simpleserial_read(cmd='r', timeout=100)
                assert bytes(response) == expected, f'got: {bytes(response)}; expected: {expected}'
            except TargetIOError as e:
                response = e.data
                #assert response is not None, f'i={i}'
                #assert response.startswith(b'set_alg ok: 3'), f'i={i}; got: {bytes(response)}'
                diff_idx = get_index_of_first_diff_bytes(response, expected)
                if diff_idx is None:
                    assert response == expected
                else:
                    assert response != expected
                if response == expected:
                    self.__logger.info('We noticed a transmission error but actually the data is fine; error in packet header / trailer?; be worried if this occurs often')
                else:
                    dist[diff_idx] += 1;
        return dist

    def simpleserial_read(self, cmd=None, pay_len=None, end='\n', timeout=250, ack=True):
        self.__handler.reset()
        ret = super(SimpleSerial2Dilithium, self).simpleserial_read(cmd=cmd, pay_len=pay_len, end=end, timeout=timeout, ack=ack)
        if self.__handler.warning_or_higher_logged:
            # let us somehow classify these error ffs -,-
            if len(self.__handler.records_warning_or_higher) == 1 and self.__handler.records_warning_or_higher[0].msg == 'Read timed out: ':
                raise TargetTimeoutError()
            raise TargetIOError(f'target logger logged a warning during simpleserial_read: {[r.msg for r in self.__handler.records_warning_or_higher]}', ret)
        return ret

    def simpleserial_cmd_until_success(self, cmd, scmd, data, cmd_read=None, pay_len=None, end='\n', timeout=250, ack=True, max_num_retries : int = None):
        num_retries = 0
        while max_num_retries is None or num_retries < max_num_retries:
            try:
                self.send_cmd(cmd, scmd, data)
                ret = self.simpleserial_read(cmd=cmd_read, pay_len=pay_len, end=end, timeout=timeout, ack=ack)
                return ret
            except TargetIOError as e:
                self.__logger.info(f'got an BlockingIOError exception: {e}; trying again ...')
                num_retries += 1
                self.flush()
        raise TargetIOError(f'Giving up reading from target after {num_retries} failed attempts.');

    def filter_msgs_one_iter(self, messages: [bytes], threshold: int = 700):
        good_messages = []
        for message in messages:
            try:
                self.sign(message, timeout=threshold)
                good_messages += [message]
            except TargetTimeoutError as e:
                print(e)
                print('Continuing, all fine ...')
                self.reboot_flush()
        return good_messages

    def __init__(self, scope = None, algorithm: int = 2, secret_key: bytes = None):
        super().__init__()

        self.__scope = None
        self.__algorithm = algorithm
        self.algorithm = algorithm
        self.__d = Dilithium(self.algorithm)
        # self.__crypto_bytes = dilithium_params[self.__algorithm]['CRYPTO_SERETKEYBTES']

        if secret_key is None:
            self.__d.pseudorandombytes_seed(b'attack-shuffling-countermeasure-keypair')
            public_key, secret_key = self.__d.keypair()
        self.__secret_key = secret_key
        # self.secret_key = secret_key

        if scope is not None:
            self.__scope.default_setup()

        self.__logger = logging.getLogger('SimpleSerial2Dilithium')
        self.__logger.setLevel(logging.NOTSET) # does this actually do something?
        self.__logger.debug('SimpleSerial2Dilithium logger says hello!')

        self.__handler = SimpleSerial2Dilithium.__handler
        self.__target_logger = logging.getLogger("ChipWhisperer Target")
        if self.__handler not in self.__target_logger.handlers:
            self.__target_logger.addHandler(self.__handler)
        self.__handler.setLevel(logging.NOTSET)


