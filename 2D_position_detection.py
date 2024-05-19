import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm callback để vẽ các điểm trên màn hình khi người dùng nhấp chuột
def mouse_drawing(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append((x, y))
# Mở video stream từ camera IP
video = cv2.VideoCapture('http://192.168.150.180:8080/video')#http://192.168.2.7:8080/: đây là IP của Camera
# Tạo cửa sổ để hiển thị video và gắn callback để vẽ các điểm khi chuột được nhấp
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)

# Mảng chứa các điểm được vẽ
circles = []

# Kích thước của hệ quy chiếu thực tế (đơn vị: mm)
height = 210
width = 297

# Kernel để sử dụng trong các phép biến đổi hình thái học
kernel = np.ones((2, 2), np.uint8)

# Vòng lặp để xử lý từng frame trong video
while True:
    ret, frame = video.read()
    frame = cv2.resize(frame,(600, 400))

    # Vẽ các điểm đã được chọn trên màn hình
    for circle in circles:
        cv2.circle(frame, circle, 2, (0,0, 255), -1)
        # print(circles)


    # Nếu đã có đủ 4 điểm, thực hiện phép biến đổi perspective để chuyển đổi ảnh
    if len(circles) == 4:
        # Tính ma trận biến đổi perspective
        pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
        pts2 = np.float32([[0, 0], [540, 0], [0, 420], [540, 420]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # Áp dụng phép biến đổi perspective lên ảnh
        object_frame = cv2.warpPerspective(frame, matrix, (540, 420))
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # i =int(input())
            image = f'./file/0lux_{i}.jpg'
            i+=1
            print(image)
            cv2.imwrite(image, frame1)

        # # Chuyển ảnh sang ảnh grayscale và làm mờ để chuẩn bị cho việc xử lý contour
        # gray = cv2.cvtColor(object_frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # gray = cv2.medianBlur(gray, 3)

        # # Áp dụng ngưỡng để tạo ảnh nhị phân
        # ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)        
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
        # thresh = cv2.dilate(thresh, kernel, iterations=5)

        # # Tìm các contours trong ảnh
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Nếu có contour được tìm thấy, xác định bounding box và tâm của contour
        # if len(contours)!= 0:
        #     largest_contour = max(contours, key=cv2.contourArea)
        #     x, y, w, h = cv2.boundingRect(largest_contour)
        #     center = (x + w / 2, y + h / 2)
        #     cv2.polylines(object_frame, [largest_contour], True, (0, 255, 0), 2)

        #     # Tham chiếu tâm của vật lên ảnh
        #     real_position = (center[0] * width / 600, center[1] * height / 400)
        #     real_position = (round(real_position[0] / 10, 2), round(real_position[1] / 10, 2))
        #     center = np.intp(center)
        #     print(real_position)
        #     # In tâm của vật
        #     cv2.putText(object_frame, "({}, {})".format(real_position[0], real_position[1]),
        #                   center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        #     # Tìm ma trận nghịch đảo của ma trận biến đổi perspective
        #     inverse_matrix = np.linalg.inv(matrix)
        #     bounding_box_object_frame = [(x, y), (x + w, y), (x, y + h), (x + w, y + h), (x + w / 2, y + h / 2)]
        #     # print(bounding_box_object_frame)
        #     # Áp dụng ma trận nghịch đảo lên tọa độ của bounding box để chuyển chúng về tọa độ của ảnh gốc
        #     new_pts = []
        #     for pt in bounding_box_object_frame:  # Đây là tọa độ của bounding box trên ảnh object_frames
        #         pt = np.array([pt], dtype='float32')
        #         pt = np.array([pt])
        #         new_pt = cv2.perspectiveTransform(pt, inverse_matrix)
        #         new_pts.append(new_pt[0][0])

        #     # Chuyển đổi tọa độ của các điểm bounding box về định dạng integer
        #     pt1 = (int(new_pts[0][0]), int(new_pts[0][1]))
        #     pt2 = (int(new_pts[3][0]), int(new_pts[3][1]))

        #     # Vẽ bounding box trên ảnh gốc
        #     frame_with_boxes = frame.copy ()
        #     cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        #     cv2.putText(frame, "({}, {})".format(real_position[0], real_position[1]),
        #                   np.intp(new_pts[4]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Hiển thị ảnh kết quả
        cv2.imshow('object_frame', object_frame)

    # Hiển thị ảnh gốc với các điểm được vẽ
    cv2.imshow('Frame', frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break

# Giải phóng tất cả các cửa sổ
video.release ()
cv2.destroyAllWindows ()