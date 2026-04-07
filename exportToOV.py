import openvino as ov

# Convert ONNX models
encModel = ov.convert_model("./onnx_exports/encoder.onnx")
decModel = ov.convert_model("./onnx_exports/decoder.onnx")

# Save them to disk in OpenVINO IR format
ov.save_model(encModel, "./openvino_exports/encoder.xml")
ov.save_model(decModel, "./openvino_exports/decoder.xml")
