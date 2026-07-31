import h5py


class NewEvaluator:
    def __init__(self, scheme):
        self.backend = scheme.backend
        self.params = scheme.params
        self.io_mode = self.params.get_io_mode()
        self.keys_path = self.params.get_keys_path()
        self.loaded_rotation_keys = set()
        self.new_evaluator()
        if self.io_mode in ("none", "save"):
            self._initialize_power_of_two_rotation_keys()

    def new_evaluator(self):
        self.backend.NewEvaluator()

    def add_rotation_key(self, amount: int):
        self.backend.AddRotationKey(amount)

    def _power_of_two_rotations(self):
        amount = 1
        while amount <= self.params.get_slots():
            yield amount
            amount *= 2

    def _galois_element(self, amount):
        return int(self.backend.GetRotationGaloisElement(amount))

    def _initialize_power_of_two_rotation_keys(self):
        if self.io_mode == "none":
            for amount in self._power_of_two_rotations():
                self.add_rotation_key(amount)
            return

        with h5py.File(self.keys_path, "a") as keys:
            for amount in self._power_of_two_rotations():
                galois_element = self._galois_element(amount)
                key_name = str(galois_element)
                if key_name in keys:
                    continue
                serialized, pointer = self.backend.GenerateAndSerializeRotationKey(
                    galois_element
                )
                try:
                    keys.create_dataset(key_name, data=serialized)
                finally:
                    self.backend.FreeCArray(pointer)

    def _load_rotation_key(self, amount):
        galois_element = self._galois_element(amount)
        if galois_element in self.loaded_rotation_keys:
            return
        with h5py.File(self.keys_path, "r") as keys:
            serialized = keys[str(galois_element)][()]
        self.backend.LoadRotationKey(serialized, galois_element)
        self.loaded_rotation_keys.add(galois_element)

    def preload_power_of_two_rotation_keys(self):
        if self.io_mode == "load":
            for amount in self._power_of_two_rotations():
                self._load_rotation_key(amount)

    def negate(self, ctxt):
        return self.backend.Negate(ctxt)
    
    def rotate(self, ctxt, amount, in_place):
        if self.io_mode == "load":
            self._load_rotation_key(amount)
        if in_place:
            return self.backend.Rotate(ctxt, amount)
        return self.backend.RotateNew(ctxt, amount)

    def add_scalar(self, ctxt, scalar, in_place):
        if in_place:
            return self.backend.AddScalar(ctxt, float(scalar))
        return self.backend.AddScalarNew(ctxt, float(scalar))

    def sub_scalar(self, ctxt, scalar, in_place):
        if in_place:
            return self.backend.SubScalar(ctxt, float(scalar))
        return self.backend.SubScalarNew(ctxt, float(scalar))

    def mul_scalar(self, ctxt, scalar, in_place):
        if isinstance(scalar, float) and scalar.is_integer():
            scalar = int(scalar)  # (e.g., 1.00 -> 1)

        if isinstance(scalar, int):
            ct_out = (self.backend.MulScalarInt if in_place 
                      else self.backend.MulScalarIntNew)(ctxt, scalar)
        else:
            ct_out = (self.backend.MulScalarFloat if in_place 
                      else self.backend.MulScalarFloatNew)(ctxt, scalar)
            ct_out = self.backend.Rescale(ct_out)

        return ct_out
        
    def add_plaintext(self, ctxt, ptxt, in_place):
        if in_place:
            return self.backend.AddPlaintext(ctxt, ptxt) 
        return self.backend.AddPlaintextNew(ctxt, ptxt) 

    def sub_plaintext(self, ctxt, ptxt, in_place):
        if in_place:
            return self.backend.SubPlaintext(ctxt, ptxt) 
        return self.backend.SubPlaintextNew(ctxt, ptxt) 

    def mul_plaintext(self, ctxt, ptxt, in_place):
        if in_place: # ct_out = ctxt
            ct_out = self.backend.MulPlaintext(ctxt, ptxt)
        else:
            ct_out = self.backend.MulPlaintextNew(ctxt, ptxt) 
        
        return self.backend.Rescale(ct_out)

    def add_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place:
            return self.backend.AddCiphertext(ctxt0, ctxt1)
        return self.backend.AddCiphertextNew(ctxt0, ctxt1)

    def sub_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place:
            return self.backend.SubCiphertext(ctxt0, ctxt1)
        return self.backend.SubCiphertextNew(ctxt0, ctxt1)

    def mul_ciphertext(self, ctxt0, ctxt1, in_place):
        if in_place: # ct_out = ctxt
            ct_out = self.backend.MulRelinCiphertext(ctxt0, ctxt1)
        else:
            ct_out = self.backend.MulRelinCiphertextNew(ctxt0, ctxt1)
        
        return self.backend.Rescale(ct_out)
    
    def rescale(self, ctxt, in_place):
        if in_place:
            return self.backend.Rescale(ctxt)
        return self.backend.RescaleNew(ctxt)
    
    def get_live_plaintexts(self):
        return self.backend.GetLivePlaintexts() 

    def get_live_ciphertexts(self):
        return self.backend.GetLiveCiphertexts()
