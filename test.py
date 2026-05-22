from pyzktorch import setup, prove, verify, verify_public_inputs
import time

config = "config.yaml"

# ---- SETUP ----
t0 = time.perf_counter()
setup(config)
t1 = time.perf_counter()
print(f"setup() terminé en {t1 - t0:.4f} secondes\n")

# ---- PROVE ----
t0 = time.perf_counter()
prove(config)
t1 = time.perf_counter()
print(f"prove() terminé en {t1 - t0:.4f} secondes\n")

# ---- VERIFY ----
t0 = time.perf_counter()
verify(config)
t1 = time.perf_counter()
print(f"verify() terminé en {t1 - t0:.4f} secondes\n")

print(f"real input verif:")
print(verify_public_inputs(config, "sample.json"))

print(f"fake input verif:")
print(verify_public_inputs(config, "sample_fake.json"))