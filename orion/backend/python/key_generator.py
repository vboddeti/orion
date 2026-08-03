import h5py

class NewKeyGenerator:
    def __init__(self, scheme):
        self.backend = scheme.backend
        self.io_mode = scheme.params.get_key_io_mode()
        self.keys_path = scheme.params.get_keys_path()
        self.sk_path = scheme.params.get_sk_path()
        self.load_secret_key = scheme.params.get_load_secret_key()
        self.new_key_generator()

    def new_key_generator(self):
        self.backend.NewKeyGenerator()
        self.generate_secret_key()
        self.generate_public_key()
        self.generate_relinearization_key()
        self.generate_evaluation_keys()

    def generate_secret_key(self):
        if self.io_mode == "load":
            # Server side (load_secret_key=False): the secret key is never
            # loaded or used — only the serialized evaluation keys are.
            if not self.load_secret_key:
                return
            with h5py.File(self.sk_path, "r") as f:
                self.backend.LoadSecretKey(f["sk"][()])
        else:
            # "none" or "save": generate a fresh secret key.
            self.backend.GenerateSecretKey()
            if self.io_mode == "save":
                sk_serial, _ = self.backend.SerializeSecretKey()
                with h5py.File(self.sk_path, "a") as f:
                    if "sk" in f:
                        del f["sk"]
                    f.create_dataset("sk", data=sk_serial)

    def generate_public_key(self):
        # The public key is used for encryption (never the secret key). It is
        # serialized in save mode and loaded in load mode, so a client can
        # encrypt with only the public key.
        if self.io_mode == "load":
            with h5py.File(self.keys_path, "r") as f:
                self.backend.LoadPublicKey(f["pk"][()])
        else:
            self.backend.GeneratePublicKey()
            if self.io_mode == "save":
                pk_serial, _ = self.backend.SerializePublicKey()
                with h5py.File(self.keys_path, "a") as f:
                    if "pk" in f:
                        del f["pk"]
                    f.create_dataset("pk", data=pk_serial)

    def generate_relinearization_key(self):
        if self.io_mode == "load":
            # Load the serialized relin key (no secret key required). The Go
            # backend rebuilds the evaluation-key set from it.
            with h5py.File(self.keys_path, "r") as f:
                self.backend.LoadRelinearizationKey(f["rk"][()])
        else:
            self.backend.GenerateRelinearizationKey()
            if self.io_mode == "save":
                rk_serial, _ = self.backend.SerializeRelinearizationKey()
                with h5py.File(self.keys_path, "a") as f:
                    if "rk" in f:
                        del f["rk"]
                    f.create_dataset("rk", data=rk_serial)

    def generate_evaluation_keys(self):
        # In load mode the evaluation-key set was already built from the loaded
        # relin key (see LoadRelinearizationKey); otherwise build it here.
        if self.io_mode != "load":
            self.backend.GenerateEvaluationKeys()

    def generate_rotation_keys(self, galois_elements):
        """Generate and persist the exact circuit rotation-key set."""
        if self.io_mode != "save":
            raise ValueError(
                "Circuit rotation keys can only be generated in key_io_mode=save"
            )
        with h5py.File(self.keys_path, "a") as keys:
            for galois_element in sorted(set(map(int, galois_elements))):
                name = str(galois_element)
                if name in keys:
                    continue
                serialized, pointer = self.backend.GenerateAndSerializeRotationKey(
                    galois_element
                )
                try:
                    keys.create_dataset(name, data=serialized)
                finally:
                    self.backend.FreeCArray(pointer)
