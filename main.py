from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.responses import JSONResponse, StreamingResponse
import io
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import pickle
import os
from typing import Dict
from io import BytesIO
from datetime import datetime, time, timezone, timedelta

from PIL import Image, ImageTk
import cv2
import util
from Silent_Face_Anti_Spoofing_master import test
from test import test

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from bson.binary import Binary
from base64 import b64encode

from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_SERVER = os.getenv("DB_SERVER")
DB_DATABASE = os.getenv("DB_DATABASE")
MONGO_URI = os.getenv("MONGO_URI")
CLIENT_URL = os.getenv("CLIENT_URL")
LOG_PATH = os.getenv("LOG_PATH")
DB_DIR = os.getenv("DB_DIR")

if not all([DB_USER, DB_PASS, DB_SERVER, DB_DATABASE, CLIENT_URL]):
    raise ValueError("Một số thông tin cấu hình không được cung cấp trong file .env.")

app = FastAPI()

# Cấu hình middleware CORS
origins = [CLIENT_URL, "http://127.0.0.1:3001"]  
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình kết nối MongoDB
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_DATABASE]
oncall_collection = db["oncallschedules"]
user_collection = db["users"]
open_attendance_collection = db["openattendances"]

# Đường dẫn để lưu embeddings
db_dir = DB_DIR
if not os.path.exists(db_dir):
    os.mkdir(db_dir)
log_path = LOG_PATH
TEMP_IMAGE_PATH = "temp_image.jpg"

@app.post("/face-register")
async def register_new_user(image: UploadFile, userId: str = Form(...)):
    # Kiểm tra định dạng tệp ảnh
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận ảnh định dạng JPEG hoặc PNG.")

    # Đọc nội dung ảnh
    img_bytes = await image.read()
    img = face_recognition.load_image_file(BytesIO(img_bytes))

    # Phát hiện và mã hóa khuôn mặt
    try:
        embeddings = face_recognition.face_encodings(img)[0]
    except IndexError:
        raise HTTPException(status_code=400, detail="Không tìm thấy khuôn mặt trong ảnh.")

    # Lưu embeddings vào tệp
    user_file = os.path.join(db_dir, f"{userId}.pickle")
    with open(user_file, "wb") as file:
        pickle.dump(embeddings, file)

    try:
        user = await user_collection.find_one({"_id": userId})
        if not user:
            raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")

        images = user.get("images", [])
        if len(images) >= 5:
            images.pop(0)

        images.append({"data": img_bytes, "timestamp": datetime.utcnow()})
        await user_collection.update_one(
            {"_id": userId},
            {"$set": {"images": images}}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu dữ liệu: {str(e)}")


    return JSONResponse(
        status_code=200,
        content={"message": f"Đăng ký thành công cho giảng viên {userId}"}
    )

@app.get("/user-image/{userId}")
async def get_user_image(userId: str):
    user = await user_collection.find_one({"_id": userId})
    if not user or "image" not in user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng hoặc ảnh.")

    try:
        vietnam_tz = timezone(timedelta(hours=7))
        images = user["images"]
        sorted_images = sorted(images, key=lambda x: x["timestamp"], reverse=True)
        response = [
            {
                "image": f"data:image/jpeg;base64,{b64encode(image['data']).decode()}",
                "timestamp": image["timestamp"]
                .replace(tzinfo=timezone.utc)
                .astimezone(vietnam_tz)  
                .isoformat() 
            }
            for image in sorted_images
        ]
        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc ảnh: {str(e)}")

async def process_login(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return {"status": "error", "message": "Không thể đọc hình ảnh"}

    # Kiểm tra chống giả mạo
    label = test(
        image_name=image,
        model_dir='./Silent_Face_Anti_Spoofing_master/resources/anti_spoof_models',
        device_id=0
    )

    print(f"Label returned from test: {label}")
    print(f"Image path: {image_path}")

    if label != 1:
        return {"status": "error", "message": "Nhận dạng thất bại. Vui lòng thử lại !"}

    # Nhận diện khuôn mặt
    userId = util.recognize(image, db_dir)
    print(f"Recognized userId: {userId}")

    if userId in ['unknown_person', 'no_persons_found']:
        return {"status": "error", "message": "Không tìm thấy người dùng"}
    else:
        user_info = await user_collection.find_one({"_id": userId})
        fullName = user_info["fullName"]

        current_date = datetime.now().date()
        current_date_datetime = datetime.combine(current_date, datetime.min.time())
        checkin_time = datetime.now()

        print(f"Checkin time: {checkin_time}")

        try:
            open_attendance = await open_attendance_collection.find_one({
                "startDay": {"$lte": current_date_datetime}, 
                "endDay": {"$gte": current_date_datetime},   
                "statusId": 4                       
            })
            if not open_attendance:
                return {"status": "error", "message": "Không tìm thấy thông tin lịch trực."}

            # Định nghĩa khung giờ trực
            morning_start = datetime.combine(current_date, datetime.strptime(open_attendance["time_In_S"], "%H:%M").time())
            morning_end = datetime.combine(current_date, datetime.strptime(open_attendance["time_Out_S"], "%H:%M").time())
            afternoon_start = datetime.combine(current_date, datetime.strptime(open_attendance["time_In_C"], "%H:%M").time())
            afternoon_end = datetime.combine(current_date, datetime.strptime(open_attendance["time_Out_C"], "%H:%M").time())
            morning_no_checkin = morning_start + timedelta(minutes=180)  
            no_checkout_morning = morning_end

            afternoon_no_checkin = afternoon_start + timedelta(minutes=180)  

            if morning_start <= checkin_time < morning_end:
                session = "S" 
            elif afternoon_start <= checkin_time:
                session = "C"
            else:
                return {"status": "error", "message": "Không nằm trong khung giờ trực hợp lệ"}

            query_schedule = {
                "userID": userId,
                "date": {
                    "$gte": datetime.combine(current_date, datetime.min.time()),  
                    "$lt": datetime.combine(current_date + timedelta(days=1), datetime.min.time())  
                },
                "onCallSession": session
            }
            schedule_record = await oncall_collection.find_one(query_schedule)

            if not schedule_record:
                return {"status": "error", "message": "Không tìm thấy lịch trực."}

            if schedule_record:
                if not schedule_record.get("attendance"):
                    if session == "S" and checkin_time <= morning_no_checkin:
                        # Cập nhật giờ check-in buổi sáng
                        await oncall_collection.update_one(
                            {"_id": schedule_record["_id"]},
                            {
                                "$set": {
                                    "attendance": True,
                                    "checkinTime": checkin_time
                                }
                            }
                        )
                    elif session == "C" and checkin_time <= afternoon_no_checkin:
                        # Cập nhật giờ check-in buổi chiều
                        await oncall_collection.update_one(
                            {"_id": schedule_record["_id"]},
                            {
                                "$set": {
                                    "attendance": True,
                                    "checkinTime": checkin_time
                                }
                            }
                        )
                    else:
                        return {
                            "status": "error",
                            "message": f"Không thể check-in sau {morning_no_checkin.strftime('%H:%M')} (sáng) hoặc {afternoon_no_checkin.strftime('%H:%M')} (chiều)"
                        }
                else:
                    if(session == "S" and checkin_time < morning_end):
                        # Nếu đã chấm công trước đó, cập nhật giờ check-out
                        await oncall_collection.update_one(
                            {"_id": schedule_record["_id"]},
                            {
                                "$set": {
                                    "checkoutTime": checkin_time
                                }
                            }
                        )
                    else:
                        return {"status": "error", "message": f"Không thể check-out sau {morning_end.strftime('%H:%M')}"}
            else:
                return {"status": "error", "message": "Bạn không có lịch trực vào buổi này !"}

            # Ghi log
            with open(log_path, 'a') as f:
                f.write(f'{userId},{checkin_time},{session}\n')

            return {"status": "success", "message": f"{fullName} đã điểm danh."}
        except Exception as e:
            print(f"Error during MongoDB operation: {e}")
            return {"status": "error", "message": str(e)}

@app.post("/oncall-check")
async def login_api(file: UploadFile = File(...)):
    try:
        # Lưu file ảnh từ request vào tạm
        with open(TEMP_IMAGE_PATH, "wb") as temp_file:
            temp_file.write(await file.read())

        # Gọi hàm xử lý đăng nhập
        result = await process_login(TEMP_IMAGE_PATH)

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

    finally:
        # Xóa file tạm sau khi xử lý
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)

    return JSONResponse(content=result)

# $env:PYTHONPATH="C:\School Stuff\HK241\production\face_id;C:\School Stuff\HK241\production\face_id\Silent_Face_Anti_Spoofing_master;$env:PYTHONPATH"
# python -m uvicorn main:app --reload      