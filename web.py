import base64
import os
import shutil
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from starlette.responses import HTMLResponse, JSONResponse
import subprocess
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()
result_path = Path("data")
style_path = Path("style")
# 设置静态文件目录
app.mount("/data", StaticFiles(directory=result_path), name="result")
app.mount("/style", StaticFiles(directory=style_path), name="style")


@app.get("/", response_class=HTMLResponse)
async def read_html():
    # 读取本地的 HTML 文件并返回
    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return html_content


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    # 保存文件到content目录下
    with open("data/content.jpg", "wb") as f:
        f.write(contents)
    return {"filename": file.filename}


@app.post("/uploadfile2/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    # 保存文件到content目录下
    with open("data/style.jpg", "wb") as f:
        f.write(contents)
    return {"filename": file.filename}


@app.post("/process_image/")
async def process_image(request: Request):
    request_body = await request.json()
    content_image = request_body.get("contentImage", "")
    style_image = request_body.get("styleImage", "")
    if content_image:
        save_base64_image(content_image, "data/content.jpg")
    if 'http' in style_image:
        style_image = '/'.join(style_image.split('/')[3:])
        shutil.copy2(style_image, "data/style.jpg")
    else:
        save_base64_image(style_image, "data/style.jpg")
    cv2.imwrite("data/content.jpg", cv2.imread("data/content.jpg"))
    cv2.imwrite("data/style.jpg", cv2.imread("data/style.jpg"))
    file_name = "./data/{}.jpg".format(int(time.time()))
    subprocess.run(["python", "trans.py", "data/content.jpg", "data/style.jpg", file_name])
    return {"result_image_path": file_name}


@app.get("/get_style_images")
async def get_style_images():
    style_images = []
    styles_dir = "./style"

    for filename in os.listdir(styles_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            style_images.append(filename)

    return JSONResponse(content={"styleImages": style_images})


def save_base64_image(base64_data, output_path):
    image_data = base64.b64decode(base64_data.split(',')[-1])

    # 将字节数据转换为NumPy数组
    np_array = np.frombuffer(image_data, np.uint8)

    # 使用OpenCV加载图像
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    cv2.imwrite(output_path, image)