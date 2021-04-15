#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import math
import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import crypten
import crypten.communicator as comm
import torch
import torch.nn.functional as F
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.tensor_types import is_float_tensor
from crypten.common.util import pool2d_reshape
from crypten.mpc import MPCTensor, ptype as Ptype
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor


class TestActive(object):
    """
    This class tests all functions of MPCTensor.
    """

    def _get_random_test_tensor(self, *args, **kwargs):
        return get_random_test_tensor(device=self.device, *args, **kwargs)

    def _check(self, encrypted_tensor, reference, msg, dst=None, tolerance=None):

        if tolerance is None:
            tolerance = getattr(self, "default_tolerance", 0.05)
        tensor = encrypted_tensor.get_plain_text(dst=dst)
        if dst is not None and dst != self.rank:
            self.assertIsNone(tensor)
        else:
            # Check sizes match
            self.assertTrue(tensor.size() == reference.size(), msg)

            self.assertTrue(is_float_tensor(reference), "reference must be a float")

            if tensor.device != reference.device:
                tensor = tensor.cpu()
                reference = reference.cpu()

            diff = (tensor - reference).abs_()
            norm_diff = diff.div(tensor.abs() + reference.abs()).abs_()
            test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
            test_passed = test_passed.gt(0).all().item() == 1
            if not test_passed:
                logging.info(msg)
                logging.info("Result %s" % tensor)
                logging.info("Reference %s" % reference)
                logging.info("Result - Reference = %s" % (tensor - reference))
            self.assertTrue(test_passed, msg=msg)
            self.assertTrue(hasattr(encrypted_tensor, "_mac"), "_mac attribute failed to propagate")

        if dst is None or dst == self.rank:
            tensor = encrypted_tensor._tensor.reveal(dst=dst)
            alpha = MPCTensor.alpha.reveal(dst=dst)
            mac = encrypted_tensor._mac.reveal(dst=dst)
            self.assertTrue((alpha * tensor == mac).all(), msg=f"{msg} - Mac Arithmetic Mismatch")
        elif dst is not None:
            self.assertIsNone(encrypted_tensor._tensor.reveal(dst=dst))
            self.assertIsNone(MPCTensor.alpha.reveal(dst=dst))
            self.assertIsNone(encrypted_tensor._mac.reveal(dst=dst))

    def test_encrypt_decrypt(self):
        """
        Tests tensor encryption and decryption for both positive
        and negative values.
        """
        sizes = [
            (),
            (1,),
            (5,),
            (1, 1),
            (1, 5),
            (5, 1),
            (5, 5),
            (1, 5, 5),
            (5, 1, 5),
            (5, 5, 1),
            (5, 5, 5),
            (1, 3, 32, 32),
            (5, 3, 32, 32),
        ]
        for size in sizes:

            # encryption and decryption without source:
            reference = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(reference)
            self._check(encrypted_tensor, reference, "en/decryption failed")
            for dst in range(self.world_size):
                self._check(
                    encrypted_tensor, reference, "en/decryption failed", dst=dst
                )

            # encryption and decryption with source:
            for src in range(self.world_size):
                input_tensor = reference if src == self.rank else []
                encrypted_tensor = MPCTensor(input_tensor, src=src, broadcast_size=True)
                for dst in range(self.world_size):
                    self._check(
                        encrypted_tensor,
                        reference,
                        "en/decryption with broadcast_size failed",
                        dst=dst,
                    )

            # test creation via new() function:
            encrypted_tensor2 = encrypted_tensor.new(reference)
            self.assertIsInstance(
                encrypted_tensor2, MPCTensor, "new() returns incorrect type"
            )
            self._check(encrypted_tensor2, reference, "en/decryption failed")

        # MPCTensors cannot be initialized with None:
        with self.assertRaises(ValueError):
            _ = MPCTensor(None)

    def test_arithmetic(self):
        """Tests arithmetic functions on encrypted tensor."""
        arithmetic_functions = ["add", "add_", "sub", "sub_", "mul", "mul_"]
        for func in arithmetic_functions:
            if func in ["mul", "mul_"]:
                continue
            for tensor_type in [lambda x: x, MPCTensor]:
                tensor1 = self._get_random_test_tensor(is_float=True)
                tensor2 = self._get_random_test_tensor(is_float=True)
                encrypted = MPCTensor(tensor1)
                encrypted2 = tensor_type(tensor2)

                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted, func)(encrypted2)

                self._check(
                    encrypted_out,
                    reference,
                    "%s %s failed"
                    % ("private" if tensor_type == MPCTensor else "public", func),
                )
                if "_" in func:
                    # Check in-place op worked
                    self._check(
                        encrypted,
                        reference,
                        "%s %s did not modify input as expected"
                        % ("private" if tensor_type == MPCTensor else "public", func),
                    )
                else:
                    # Check original is not modified
                    self._check(
                        encrypted,
                        tensor1,
                        "%s %s modified input"
                        % ("private" if tensor_type == MPCTensor else "public", func),
                    )

                # Check encrypted vector with encrypted scalar works.
                tensor1 = self._get_random_test_tensor(is_float=True)
                tensor2 = self._get_random_test_tensor(is_float=True, size=(1,))
                encrypted1 = MPCTensor(tensor1)
                encrypted2 = MPCTensor(tensor2)
                reference = getattr(tensor1, func)(tensor2)
                encrypted_out = getattr(encrypted1, func)(encrypted2)
                self._check(encrypted_out, reference, "private %s failed" % func)

        # test square
        # tensor = self._get_random_test_tensor(is_float=True)
        # reference = tensor * tensor
        # encrypted = MPCTensor(tensor)
        # encrypted_out = encrypted.square()
        # self._check(encrypted_out, reference, "square failed")

        # # Test radd, rsub, and rmul
        # reference = 2 + tensor1
        # encrypted = MPCTensor(tensor1)
        # encrypted_out = 2 + encrypted
        # self._check(encrypted_out, reference, "right add failed")

        # reference = 2 - tensor1
        # encrypted_out = 2 - encrypted
        # self._check(encrypted_out, reference, "right sub failed")

        # reference = 2 * tensor1
        # encrypted_out = 2 * encrypted
        # self._check(encrypted_out, reference, "right mul failed")

    def test_div(self):
        """Tests division of encrypted tensor by scalar and tensor."""
        for function in ["div", "div_"]:
            raise NotImplementedError
            for scalar in [2, 2.0]:
                tensor = self._get_random_test_tensor(is_float=True)

                reference = tensor.float().div(scalar)
                encrypted_tensor = MPCTensor(tensor)
                encrypted_tensor = getattr(encrypted_tensor, function)(scalar)
                self._check(encrypted_tensor, reference, "scalar division failed")

                # multiply denominator by 10 to avoid dividing by small num
                divisor = self._get_random_test_tensor(is_float=True, ex_zero=True) * 10
                reference = tensor.div(divisor)
                encrypted_tensor = MPCTensor(tensor)
                encrypted_tensor = getattr(encrypted_tensor, function)(divisor)
                self._check(encrypted_tensor, reference, "tensor division failed")

    def test_copy_clone(self):
        """Tests shallow_copy and clone of encrypted tensors."""
        sizes = [(5,), (1, 5), (5, 10, 15)]
        for size in sizes:
            tensor = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor = MPCTensor(tensor)

            # test shallow_copy
            encrypted_tensor_shallow = encrypted_tensor.shallow_copy()
            self.assertEqual(
                id(encrypted_tensor_shallow._tensor), id(encrypted_tensor._tensor)
            )
            self._check(encrypted_tensor_shallow, tensor, "shallow_copy failed")
            # test clone
            encrypted_tensor_clone = encrypted_tensor.clone()
            self.assertNotEqual(
                id(encrypted_tensor_clone._tensor), id(encrypted_tensor._tensor)
            )
            self._check(encrypted_tensor_clone, tensor, "clone failed")

    def test_copy_(self):
        """Tests copy_ function."""
        sizes = [(5,), (1, 5), (5, 10, 15)]
        for size in sizes:
            tensor1 = self._get_random_test_tensor(size=size, is_float=True)
            tensor2 = self._get_random_test_tensor(size=size, is_float=True)
            encrypted_tensor1 = MPCTensor(tensor1)
            encrypted_tensor2 = MPCTensor(tensor2)
            encrypted_tensor1.copy_(encrypted_tensor2)
            self._check(encrypted_tensor1, tensor2, "copy_ failed")


# Run all unit tests with both TFP and TTP providers
class TestTFP(MultiProcessTestCase, TestActive):
    def setUp(self):
        self.active_security = MPCTensor.ACTIVE_SECURITY
        MPCTensor.ACTIVE_SECURITY = True
        self._original_provider = crypten.mpc.get_default_provider()
        crypten.CrypTensor.set_grad_enabled(False)
        crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedFirstParty)
        super(TestTFP, self).setUp()

    def tearDown(self):
        MPCTensor.ACTIVE_SECURITY = self.active_security
        crypten.mpc.set_default_provider(self._original_provider)
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTFP, self).tearDown()


class TestTTP(MultiProcessTestCase, TestActive):
    def setUp(self):
        self._original_provider = crypten.mpc.get_default_provider()
        crypten.CrypTensor.set_grad_enabled(False)
        crypten.mpc.set_default_provider(crypten.mpc.provider.TrustedThirdParty)
        super(TestTTP, self).setUp()

    def tearDown(self):
        crypten.mpc.set_default_provider(self._original_provider)
        crypten.CrypTensor.set_grad_enabled(True)
        super(TestTTP, self).tearDown()


# This code only runs when executing the file outside the test harness (e.g.
# via the buck target of another test)
if __name__ == "__main__":
    unittest.main()
