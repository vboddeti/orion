import h5py


class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme
        self.backend = scheme.backend
        self.io_mode = scheme.params.get_io_mode()
        self.keys_path = scheme.params.get_keys_path()

    def __del__(self):
        self.backend.DeleteBootstrappers()

    def generate_bootstrapper(self, slots):
        # We will wait to instantiate any bootstrapper until our bootstrap
        # placement algorithm determines they're necessary.
        logp = self.scheme.params.get_boot_logp()
        if self.io_mode == "none":
            return self.backend.NewBootstrapper(logp, slots)

        if self.io_mode == "save":
            serialized, pointer = self.backend.GenerateAndSerializeBootstrapper(
                logp, slots
            )
            try:
                with h5py.File(self.keys_path, "a") as keys:
                    group = keys.require_group("bootstrappers")
                    key = str(slots)
                    if key in group:
                        del group[key]
                    group.create_dataset(key, data=serialized)
            finally:
                self.backend.FreeCArray(pointer)
            return None

        with h5py.File(self.keys_path, "r") as keys:
            serialized = keys["bootstrappers"][str(slots)][()]
        return self.backend.LoadBootstrapper(logp, slots, serialized)
    
    def bootstrap(self, ctxt, slots):
        return self.backend.Bootstrap(ctxt, slots)
