from roboflow import Roboflow



rf = Roboflow("v8ySLCA54CItwUOc9eMw")
project = rf.workspace("defectdetection-tot3d").project("mvtec-ad-defect")
version = project.version(1)

# 원하는 포맷 지정 (예: yolov8)
download_path = version.download(model_format="yolov8")  # .zip이 받아지며, 풀면 .pt 포함
print(f"다운로드된 경로: {download_path}")

