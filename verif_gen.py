from pyzktorch import verify, verify_public_inputs, verify_public_outputs

config= "v9_config/config.yaml"

verify(config)

ok_outputs = verify_public_outputs(config, "v9_config/final_output_float.json", "float")
print("Final outputs match?", ok_outputs)

ok_outputs = verify_public_outputs(config, "v9_config/final_output_float_fake.json", "float")
print("Fake final outputs match?", ok_outputs)

ok_inputs = verify_public_inputs(config, "v9_config/input_match_GWB.json")
print("Inputs match?", ok_inputs)

ok_inputs = verify_public_inputs(config, "v9_config/input_match_GWB_fake.json")
print("Fake Inputs match?", ok_inputs)