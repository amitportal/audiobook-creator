import logging
import torch
import onnxruntime as ort
try:
    import openvino.runtime as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_hardware_info():
    """
    Detect available hardware and return a prioritized list of execution providers and devices.
    """
    info = {
        "cuda": [],
        "openvino": [],
        "dml": False,
        "cpu": True
    }
    
    # 1. Check CUDA
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda"].append({
                "id": i,
                "name": props.name,
                "memory": props.total_memory / (1024**3)
            })
            
    # 2. Check OpenVINO (Intel NPU/GPU/CPU)
    if OPENVINO_AVAILABLE:
        try:
            core = ov.Core()
            devices = core.available_devices
            for dev in devices:
                info["openvino"].append(dev)
        except Exception as e:
            logger.debug(f"OpenVINO detection failed: {e}")
            
    # 3. Check ONNX Providers
    ort_providers = ort.get_available_providers()
    info["dml"] = "DmlExecutionProvider" in ort_providers
    
    return info

def select_best_device():
    """
    Select the best device based on availability.
    Priority: CUDA > OpenVINO (NPU) > OpenVINO (GPU) > DirectML > CPU
    """
    hw = get_hardware_info()
    
    if hw["cuda"]:
        # Pick GPU with most memory
        best_gpu = max(hw["cuda"], key=lambda x: x["memory"])
        return "cuda", f"cuda:{best_gpu['id']}"
    
    if "NPU" in hw["openvino"]:
        return "openvino", "NPU"
    
    if "GPU" in hw["openvino"]:
        return "openvino", "GPU"
        
    if hw["dml"]:
        return "dml", "dml"
        
    return "cpu", "cpu"

def get_onnx_providers(target_device=None):
    """Get prioritized list of ONNX execution providers"""
    available = ort.get_available_providers()
    
    if target_device is None:
        type, name = select_best_device()
    else:
        type = target_device
        
    providers = []
    
    if type == "cuda" and "CUDAExecutionProvider" in available:
        # If specific ID in name (e.g. cuda:1)
        device_id = 0
        if ":" in name:
            try:
                device_id = int(name.split(":")[1])
            except: pass
        providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        
    if type == "openvino" and "OpenVINOExecutionProvider" in available:
        # OpenVINO can target NPU, GPU, or CPU
        # Example: {"device_type": "NPU_FP16"}
        ov_device = "CPU"
        hw_info_ov = get_hardware_info()["openvino"]
        if "NPU" in hw_info_ov:
            ov_device = "NPU"
        elif "GPU" in hw_info_ov:
            ov_device = "GPU"
        
        providers.append(("OpenVINOExecutionProvider", {"device_type": ov_device}))
        
    if type == "dml" and "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
        
    providers.append("CPUExecutionProvider")
    return providers
